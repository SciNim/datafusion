class GroupedSeries:
    # Note: all constructors should either use GroupedSeries.empty or sort the groups beforehand
    def __init__(
        self,
        epochs: ArrayLike,
        groups: list[tuple[Tag, ...]],
        values: ArrayLike,
        is_interpolated: ArrayLike,
        dense_epochs: ArrayLike | None = None,
        source_info: SourceInfo | None = None,
        forecast_lower_values: ArrayLike | None = None,
        forecast_upper_values: ArrayLike | None = None,
        forecast_values: ArrayLike | None = None,
        anomalies_lower_values: ArrayLike | None = None,
        anomalies_upper_values: ArrayLike | None = None,
        anomalies_scores_values: ArrayLike | None = None,
        anomalies_ratings_values: ArrayLike | None = None,
        keep_interpolated: bool = False,
        interval: int | None = None,
        extra_info: dict[str, list[MetricsResponseInfo] | EventResponseInfo] | None = None,
    ) -> None:
        """
        Create a new `GroupedSeries` object.

        params:
        epochs: numpy int32 array of shape (epochs,)
        groups: list of groupings `[tuple(str, ...)]`.
        values: numpy array of shape (len(groups), len(epochs))
        is_interpolated: numpy bool array (same length as epochs)
        dense_epochs: numpy int32 array (dense_epochs) : optional interval-aligned list of timestamps. Unlike epochs,
            this includes timestamps which have no corresponding value.
        source_info: SourceInfo object (or None).
        forecast_lower_values: optional numpy array of shape (len(groups), len(epochs)), only filled in by forecast()
        forecast_upper_values: optional numpy array of shape (len(groups), len(epochs)), only filled in by forecast()
        forecast_values: optional numpy array of shape (len(groups), len(epochs)), only filled in by forecast()
        anomalies_lower_values: optional numpy array of shape (len(groups), len(epochs)), only filled in by anomalies()
        anomalies_upper_values: optional numpy array of shape (len(groups), len(epochs)), only filled in by anomalies()
        anomalies_scores_values: optional numpy array of shape (len(groups), len(epochs)), only filled in by anomalies()
        anomalies_ratings_values: optional numpy array of shape (len(groups), len(epochs)),only filled in by anomalies()
        interval: int : interval we get back from the querier indicating which value was used for the query
        """
        self.epochs = np.asarray(epochs, dtype=np.int32)
        self.dense_epochs = None if dense_epochs is None else np.asarray(dense_epochs, dtype=np.int32)
        self.source_info = source_info if source_info is not None else SourceInfo()
        self.values = np.asarray(values, dtype=np.float64)

        self.is_interpolated = np.asarray(is_interpolated, dtype=bool)
        # invariant: the list of groups must match the order of self.values
        # NOTE: it might be nice to have an OrderedDict here, so we can map
        # group -> index and not just index -> group
        self.groups: list[tuple[Tag, ...]] = list(groups)

        if len(self.values) == 0:
            # empty arrays should still be 2D, even if (0, 0)
            self.values = self.values.reshape((0, 0))
        else:
            # Just make sure this is already the shape it should be
            assert self.values.shape == (len(self.groups),) + self.epochs.shape

        # optional values, some set by anomalies() only and some by forecast() only as per below:
        def none_if_none_else_asarray(input):
            return None if input is None else np.asarray(input, dtype=np.float64)

        # from forecast()
        self.forecast_lower_values = none_if_none_else_asarray(forecast_lower_values)
        self.forecast_upper_values = none_if_none_else_asarray(forecast_upper_values)
        self.forecast_values = none_if_none_else_asarray(forecast_values)
        # from anomalies()
        self.anomalies_lower_values = none_if_none_else_asarray(anomalies_lower_values)
        self.anomalies_upper_values = none_if_none_else_asarray(anomalies_upper_values)
        self.anomalies_scores_values = none_if_none_else_asarray(anomalies_scores_values)
        self.anomalies_ratings_values = none_if_none_else_asarray(anomalies_ratings_values)

        # optional value to NOT remove interpolated values, used for comparing
        # final values between flushing aggr queries and non aggr queries
        # (non aggr would normally be filtered if there is a time bucket that has
        # only interpolated values)
        self.keep_interpolated = keep_interpolated

        # After the migration of modifier logic to the querier, it may use a different
        # interval than what was given to it in the request so we set it here.
        self.interval = interval

        # extra info from different query sources, currently only used by metrics.
        self.extra_info = extra_info

    @classmethod
    def empty(
        cls,
        epochs: ArrayLike | None = None,
        groups: list[tuple[Tag, ...]] | None = None,
        is_interpolated: ArrayLike | None = None,
        source_info: SourceInfo | None = None,
        dense_epochs: ArrayLike | None = None,
        interval: int | None = None,
        extra_info: dict[str, list[MetricsResponseInfo] | EventResponseInfo] | None = None,
    ) -> GroupedSeries:
        if epochs is None:
            epochs = np.array([], dtype=np.int32)
        if groups is None:
            groups = []
        if is_interpolated is None:
            is_interpolated = np.zeros((len(epochs),), dtype=bool)  # type: ignore
        values = np.empty((len(groups), len(epochs)), dtype=np.float64)  # type: ignore
        # Just to be "safe"
        values.fill(np.nan)

        return cls(
            epochs,
            dense_epochs=dense_epochs,
            groups=groups,
            values=values,
            is_interpolated=is_interpolated,
            source_info=source_info,
            interval=interval,
            extra_info=extra_info,
        )

    @classmethod
    def from_series(cls, groups_to_series, source_info=None):
        """
        Takes a map of `{group: series}` where `group: tuple(str)` and `series: MetricSeries`, and converts
        it to a GroupedSeries. Assumes series are already aligned.

        groups_to_series should have correct group results - no group should
        have a key value of None. However, a MetricSeries may be empty.
        """
        # TODO: this is ugly, and either requires error-checking or a lack of safety.
        # A more efficient method would build this up from raw series
        if len(groups_to_series) == 0:
            return cls.empty(source_info=source_info)
        epochs = next(iter(groups_to_series.values())).dates().as_epoch()
        grouped_series = cls.empty(epochs, groups=groups_to_series.keys(), source_info=source_info)

        for ix, g in enumerate(grouped_series.groups):
            grouped_series.values[ix, :] = groups_to_series[g].values()
        return grouped_series

    @classmethod
    def concatenate(cls, series_to_concatenate):
        """
        Concatenate a list of GroupedSeries objects into a single GroupedSeries object.

        series-to_concatenate should have consecutive epochs - each successive one should start
        at a date the previous one left off at.

        If a group is in one GroupedSeries but not another, it will appear in the output with
        the data input, and the epochs where it was not listed will be NaN.
        """
        if len(series_to_concatenate) == 1:
            return series_to_concatenate[0]

        groups = set()
        epochs = []
        dense_epochs = None
        date_out_ixs = []

        # TODO: overlapping groupedseries means that we should still be using BatchedSeries
        # (because then we can merge points with the same epoch)
        # We don't do it yet, so for now let's remove interpolated points when it happens.
        if cls._epochs_overlap(series_to_concatenate):
            for gs in series_to_concatenate:
                gs.remove_interpolated()

        # Figure out the output groups & epochs, so we know what size array we will need
        for gs in series_to_concatenate:
            groups.update(gs.groups)
            if gs.dense_epochs is not None:
                dense_epochs = (
                    np.copy(gs.dense_epochs) if dense_epochs is None else np.union1d(dense_epochs, gs.dense_epochs)
                )
            if len(epochs) > 0 and len(gs.epochs) > 0:
                assert epochs[-1] < gs.epochs[0]
            ix_start = len(epochs)
            epochs.extend(gs.epochs)
            ix_end = len(epochs)
            date_out_ixs.append((ix_start, ix_end))

        epochs_arr = np.asarray(epochs, dtype=np.int32)
        # Create (and allocate) space for the new array
        concatenated = GroupedSeries.empty(epochs=epochs_arr, groups=sorted(groups))
        concatenated.dense_epochs = dense_epochs

        # We assume here that the intervals match for all GS objects
        if len(series_to_concatenate) > 0:
            concatenated.interval = series_to_concatenate[0].interval

        # Map of (group -> index) for the new series
        groups_to_indices = {g: n for n, g in enumerate(concatenated.groups)}

        for (dix_start, dix_end), series in zip(date_out_ixs, series_to_concatenate):
            out_group_ixs = [groups_to_indices[g] for g in series.groups]

            # And now we copy the data from each old series to the right section of the new one
            # For any group that was missing in an old series, that section will be untouched
            # for the new series, leaving it NaN
            concatenated.values[out_group_ixs, dix_start:dix_end] = series.values
            concatenated.is_interpolated[dix_start:dix_end] = series.is_interpolated
            concatenated.source_info.update(series.source_info)
        return concatenated

    @classmethod
    def _epochs_overlap(cls, gss):
        prev_end = None
        for gs in gss:
            if len(gs.epochs) == 0:
                continue
            if prev_end is None:
                prev_end = gs.epochs[-1]
            elif prev_end >= gs.epochs[0]:
                return True
        return False

    @classmethod
    def from_query_results(cls, query_results_list):
        """
        Creates a GroupedSeries from a list of ALIGNED QueryResult.
        """
        if not query_results_list:
            return GroupedSeries.empty()
        epochs = query_results_list[0].dates_epoch()
        groups_to_qr = {qr.grouping_union(): qr for qr in query_results_list}
        gs = GroupedSeries.empty(epochs=epochs, groups=groups_to_qr.keys())
        for ix, g in enumerate(gs.groups):
            gs.values[ix, :] = groups_to_qr[g].values()

        return gs

    def to_query_results(self, qp, metric_type, true_interval=None):
        """
        Convert to a list of QueryResult objects. This is a wasteful procedure, but
        QueryResult objects are what some downstream functions work with.
        """
        self.remove_interpolated()
        results = []
        interval = true_interval if true_interval is not None else qp.interval
        if qp is not None:
            with qp.freeze_hash():
                for ix, grouping in enumerate(self.groups):
                    series = MetricSeries(
                        dates=Dates(self.epochs),
                        values=self.values[ix, :],
                        dim_metric_key=qp.metric.key,
                        rollup_interval=interval,
                    )
                    result = QueryResult(
                        qp.metric_name,
                        interval=interval,
                        grouping={qp: grouping},
                        filters={qp: qp.filter_query},
                        series=series,
                        aggr=qp.aggregator,
                    )
                    result.interp_limit = qp.interp_limit
                    result.interp_method = qp.interp_method
                    result.metric_type = metric_type
                    results.append(result)
        return results

    def remove_interpolated(self):
        """
        Remove interpolated data.
        """
        if self.keep_interpolated:
            return

        if np.any(self.is_interpolated) and len(self.values) > 0:
            self.values = self.values[:, ~self.is_interpolated]
            self.epochs = self.epochs[~self.is_interpolated]
            self.is_interpolated = self.is_interpolated[~self.is_interpolated]

        if self.values.size == 0:
            self.remove_all_group_data()

    def remove_all_group_data(self):
        """
        make sure that empty means empty, and not epochs with no values

        Quirk note (TSQ-1029):
        This behavior is different from empty() which returns a new
        GroupSeries with 2d array filled with NaNs as values. See the
        comments for is_empty() for more information
        """

        self.values = np.array([], dtype=np.float64)
        self.epochs = np.array([], dtype=np.int32)
        self.is_interpolated = np.array([], dtype=bool)
        self.groups = []

    def remove_all_group_data_leaving_epochs(self):
        # same as above, but don't get rid of the epochs.
        # you should only use this function if you intend on adding group data later (self.add_group()). Otherwise use the above function.
        self.values = np.empty((0, self.num_epochs()))
        self.groups = []

    def slice_by_epoch(self, start, end):
        """
        trims the series to the requested time range [start, end)
        and removes all-nan groups.
        If the GroupedSeries object contains a FADO values matrix (anomalies or forecast),
        the trimming will also be applied to these matrices.
        """
        epochs = self.epochs
        if epochs is None or len(epochs) < 1 or (start <= epochs[0] and epochs[-1] < end):
            return self

        selected_epochs = (epochs >= start) & (epochs < end)

        nan_values = np.isnan(self.values[:, selected_epochs])
        if self.forecast_values is not None:
            # in the case of forecast, we must keep groups that have forecast values in the requested interval
            # if the requested interval is in the future, the values array will be Nan there
            nan_forecast = np.isnan(self.forecast_values[:, selected_epochs])
            np.logical_and(nan_forecast, nan_values, out=nan_values)
        # keep only groups that have at least one non-nan value or forecasted_value
        # in the requested interval [start, end)
        selected_groups = ~np.all(nan_values, axis=1)

        new_groups = itertools.compress(self.groups, selected_groups)

        def none_if_none_else_slice(array):
            return None if array is None else array[selected_groups, :][:, selected_epochs]

        new_values = none_if_none_else_slice(self.values)
        # special case to avoid empty groupedseries with non-empty epochs
        if new_values.size == 0:
            return GroupedSeries.empty()

        # apply the same slicing to all FADO values matrices
        kwargs_ds_attributes = dict()
        for attr in FADO_GROUPED_SERIES_ATTRIBUTES:
            kwargs_ds_attributes[attr] = none_if_none_else_slice(getattr(self, attr))

        new_epochs = epochs[selected_epochs]
        new_is_interpolated = self.is_interpolated[selected_epochs]

        return GroupedSeries(
            epochs=new_epochs,
            groups=new_groups,
            values=new_values,
            is_interpolated=new_is_interpolated,
            **kwargs_ds_attributes,
        )

    def monotonic_diff_values(self):
        values = self.values
        # take regular numpy diff
        diff = np.diff(values)
        # replace all nans with the "next" values in arr
        np.putmask(diff, np.isnan(diff), values[:, 1:])
        # replace all negatives with the "next" values in arr
        np.putmask(diff, diff < 0, values[:, 1:])
        # keep the original first elements
        self.values = np.concatenate([values[:, :1], diff], axis=1)

    def diff_values(self, kind=Diff.ZERO):
        """
        Diff the values for each group, zeroing the negative values or taking the
        monotonic difference if asked.
        """
        if kind == Diff.MONOTONIC:
            # Note: likely unused
            self.monotonic_diff_values()
        else:
            diff = np.diff(self.values, axis=1)
            if kind == Diff.ZERO:
                diff[diff < 0] = 0

            nans = np.empty((self.num_groups(), 1))
            nans[:] = np.nan

            self.values = np.concatenate([nans, diff], axis=1)

    def shift_epochs(self, shift_s, allow_dense_epochs=False):
        """
        Shift epochs by shift_s seconds
        """
        if self.dense_epochs is not None:
            if allow_dense_epochs:
                self.dense_epochs += shift_s
            else:
                # TODO To support this, we need to match the timeshift modifier
                # implementation in the metrics-querier
                raise QueryParsingException("Epoch shift not supported for calendar-aligned series!")

        self.epochs += shift_s

    def last_epoch(self):
        """Return last_epoch without checking for its existence"""
        return self.epochs[-1]

    def positivize(self):
        """Get rid of negative values in the series, in-place."""
        v = self.values.flat
        if len(v) == 0:
            return

        # only update the series if we need to, for perf
        neg_index = np.where(v < 0)[0]
        if len(neg_index) == 0:
            return

        # if there's only a single, negative value, set it to 0
        if len(v) == 1:
            v[0] = 0
            return

        # otherwise, just repeat the closest value
        for i in neg_index:
            if i == 0:
                v[i] = v[i + 1]
            else:
                v[i] = v[i - 1]

    def view(self, group):
        """
        Get a view of the values for a single group.
        """
        ix = self.groups.index(group)
        return self.values[ix, :]

    def view_groups(self, groups):
        """
        Return values corresponding to the given groups
        groups is expected to be a subset of self.groups
        It returns the values in the order of the given groups
        """
        groups_to_index = {group: index for index, group in enumerate(groups)}
        groups_indices = np.zeros(len(groups), dtype=int)
        found = 0
        for i, g in enumerate(self.groups):
            if g in groups_to_index:
                groups_indices[groups_to_index[g]] = i
                found += 1
                if found == len(groups):
                    break
        return self.values[groups_indices, :]

    def copy(self):
        """
        Returns an exact duplicate of the current GS
        """
        epochs = np.copy(self.epochs)
        dense_epochs = None if self.dense_epochs is None else np.copy(self.dense_epochs)
        values = np.copy(self.values)
        groups = self.groups[:]
        is_interpolated = np.copy(self.is_interpolated)
        return GroupedSeries(
            epochs=epochs,
            values=values,
            groups=groups,
            dense_epochs=dense_epochs,
            is_interpolated=is_interpolated,
            source_info=self.source_info.copy(),
            keep_interpolated=self.keep_interpolated,
            interval=self.interval,
            extra_info=self.extra_info,
        )

    def to_dataframe(self):
        """Convert to a pandas dataframe."""
        # TODO[wendell]: use something like pandas.to_datetime for this
        return pandas.DataFrame(self.values.T, index=self.datetimes(), columns=self.groups)

    def __str__(self):
        return self.to_dataframe().to_string()

    def num_groups(self):
        """Returns number of groups"""
        return len(self.groups)

    def num_epochs(self):
        """Returns number of epochs"""
        return len(self.epochs)

    def get_matrix_byte_size(self):
        """Returns the matrix byte size, along with any extra tags"""
        size = (self.num_epochs() * 4) + (self.num_epochs() * self.num_groups() * 8) + len(self.is_interpolated)
        tags = []
        if self.dense_epochs is not None:
            size += self.num_dense_epochs() * 4
            tags += ["contains_dense_epochs:true"]

        if self.anomalies_lower_values is not None:
            # four extra matrices of the same size
            size += (self.num_epochs() * self.num_groups() * 8) * 4
            tags += ["contains_anomalies:true"]
        elif self.forecast_lower_values is not None:
            # three extra matrices of the same size
            size += (self.num_epochs() * self.num_groups() * 8) * 3
            tags += ["contains_forecast:true"]

        return size, tags

    def num_dense_epochs(self):
        return 0 if self.dense_epochs is None else len(self.dense_epochs)

    def is_empty(self):
        """
        Returns the emptiness status

        Quirk note (TSQ-1029):
        We currently have 2 ways to empty out values:
            - empty() returns a new GroupSeries with a 2d array filled with NaNs as values
            - remove_all_group_data() sets values to an empty 1d array

        This function returns False when empty() is called with non-zero epochs & groups
        size. e.g. GroupedSeries.empty(10,3).is_empty() == False
        """
        return self.values.size == 0

    def datetimes(self):
        """Returns datetimes values of epochs"""
        return Dates(self.epochs).as_dt()

    def add_group(self, group_name: tuple[Tag, ...], data: ArrayLike) -> None:
        """
        Manually adds a new group to this GroupedSeries, with data
        ASSUMES that the length of the data == number of epochs.
        """
        self.groups.append(group_name)
        self.values = np.vstack([self.values, np.array(data).copy()])

    def __eq__(self, other):
        """
        Mostly for testing purposes
        """
        groups_to_index = {g: i for i, g in enumerate(self.groups)}
        order = np.zeros(len(self.groups), dtype=int)
        for i, g in enumerate(other.groups):
            if g not in groups_to_index:
                return False
            order[groups_to_index[g]] = i

        for attr in FADO_GROUPED_SERIES_ATTRIBUTES:
            self_attr = getattr(self, attr)  # type is Optional[np.array]
            other_attr = getattr(other, attr)  # type is Optional[np.array]
            if self_attr is not None or other_attr is not None:
                if self_attr is None or other_attr is None:
                    # if one GroupedSeries has the attribute, the other one needs to have it too
                    return False
                if not np.allclose(self_attr, other_attr[order, :], equal_nan=True):
                    return False

        return (
            np.array_equal(self.epochs, other.epochs)
            and sorted(self.groups) == sorted(other.groups)
            and np.allclose(self.values, other.values[order, :], equal_nan=True)
            and np.array_equal(self.is_interpolated, other.is_interpolated)
        )

    def __hash__(self):
        return hash((tuple(np.ravel(self.epochs)), tuple(sorted(self.groups)), tuple(np.ravel(self.is_interpolated))))

    def __repr__(self):
        """
        To have access to a quick summary of the content.
        """
        return (
            f"<GroupedSeries(n_epochs={self.num_epochs()},n_groups={self.num_groups()},"
            f"shape_values={str(self.values.shape)},interval={self.interval}) at {hex(id(self))}>"
        )

    @classmethod
    def serialize(cls, groupedseries):
        """Return a serialized representation of the given query results."""
        return GroupedSeriesCodec.encode(groupedseries)

    @classmethod
    def deserialize(cls, payload):
        """Deserialize the given payloads of query results."""
        epochs, values, groups, is_interpolated = GroupedSeriesCodec.decode(payload)

        return cls(epochs, values=values, groups=groups, is_interpolated=is_interpolated)

    def to_protobuf(self, include_groups=True):
        pb = MetricsBuffer()
        if include_groups:
            pb.groups[:] = ["\t".join(gs) for gs in self.groups]
        pb.num_groups = len(self.groups)
        pb.num_epochs = len(self.epochs)

        epochs_arr = self.epochs.astype(np.int64)
        pb.epochs_buf = lz4.compress(epochs_arr.tobytes())

        if self.dense_epochs is not None:
            pb.num_dense_epochs = len(self.dense_epochs)
            dense_epochs_arr = self.dense_epochs.astype(np.int64)
            pb.dense_epochs_buf = lz4.compress(dense_epochs_arr.tobytes())

        pb.has_data_buf = lz4.compress((~self.is_interpolated).tobytes())

        assert self.values.dtype == np.float64
        pb.values_buf = lz4.compress(self.values.tobytes())
        return pb

    @tracer.wrap("groupedseries.to_extra_info")
    def to_extra_info(self, metrics_data: PipelineHealthResponse) -> None:
        trace.set_tag("metrics_extra_info_present", metrics_data.ByteSize() > 0)
        if metrics_data.ByteSize() > 0:
            self.extra_info = {METRICS_RESPONSE_INFO: [MetricsResponseInfo(pipeline_health_response=metrics_data)]}
            statsd.increment(
                "dd.metrics.query.koutris_pushdown",
                tags=["direction:from_metrics"],
            )

    @classmethod
    def validate_groups_and_epochs(cls, response, groups=None, use_comma_delimiter=False):
        if groups is None:
            # Use new \t delimiter if available. This allows our system to support arbitrary tags.
            if any("\t" in group_tags for group_tags in response.groups) or not use_comma_delimiter:
                statsd.increment(
                    "dd.metrics.query.groups.use_tab_delimiter", tags=["source:groupedseries.from_protobuf"]
                )
                groups = [tuple(g.split("\t")) if g else EMPTY_GROUP for g in response.groups]
        else:
            statsd.increment("dd.metrics.query.groups.use_comma_delimiter", tags=[])
            groups = [tuple(g.split(",")) if g else EMPTY_GROUP for g in response.groups]

        assert len(groups) == response.num_groups, "Found %d groups with a reported %d groups" % (
            len(groups),
            response.num_groups,
        )
        epochs = np.frombuffer(lz4.decompress(response.epochs_buf), dtype=np.int64)
        assert len(epochs) == response.num_epochs, "Found %d epochs with a reported %d epochs" % (
            len(epochs),
            response.num_epochs,
        )
        return groups, epochs.astype(np.int32)

    @classmethod
    def get_dense_epochs(cls, response):
        dense_epochs = None
        if isinstance(response, MetricsBuffer):
            if response.num_dense_epochs > 0 and response.dense_epochs_buf:
                dense_epochs = np.frombuffer(lz4.decompress(response.dense_epochs_buf), dtype=np.int64)
                dense_epochs = dense_epochs.astype(np.int32)
                if dense_epochs is not None and len(dense_epochs) == 0:
                    dense_epochs = None
        return dense_epochs

    @classmethod
    def from_protobuf(cls, response, groups=None, source_info=None, use_comma_delimiter=False):
        """
        Takes a MetricsBuffer protobuf response and converts it to a GroupedSeries.

        Assumes `has_data_buf` is empty, and `groups` must be set either as a parameter
        or in the MetricsBuffer
        """
        if response.num_groups == 0:
            return cls.empty()

        groups, epochs = cls.validate_groups_and_epochs(response, groups, use_comma_delimiter)
        dense_epochs = cls.get_dense_epochs(response)
        gs = cls.empty(epochs, groups, dense_epochs=dense_epochs, source_info=source_info)
        # This is a read-only wrapper around the underlying buffer, no copy
        values_flat = np.frombuffer(lz4.decompress(response.values_buf), dtype=np.float64)

        # If there are counts in the metrics buffer... then we should be using a BatchedSeries.
        # This is an invalid response for converting to a GroupedSeries.
        assert not response.counts_buf

        # Reshape values_flat into the pre-allocated original array
        gs.values.flat = values_flat

        gs.is_interpolated[:] = ~np.frombuffer(lz4.decompress(response.has_data_buf), dtype=bool)

        return gs

    @classmethod
    def from_multicompute_protobuf(cls, response, multi_compute_count):
        """
        Takes a MetricsBuffer protobuf response from event platform and converts it to a GroupedSeries.

        This is a special case for the event platform, where we have multiple compute values
        """
        if response.num_groups == 0:
            return []

        groups, epochs = cls.validate_groups_and_epochs(response)
        dense_epochs = cls.get_dense_epochs(response)

        gs = cls.empty(epochs, groups, dense_epochs=dense_epochs)
        # This is a read-only wrapper around the underlying buffer, no copy
        values_flat = np.frombuffer(lz4.decompress(response.values_buf), dtype=np.float64)

        # Here is where we diverge from the normal from_protobuf method. We reshape out values_flat
        # here manually into a three dimensional array, where the first dimension is the number of groups,
        # the second dimension is the number of epochs, and the third dimension is the number of computes.
        values_reshaped = values_flat.reshape((len(groups), len(epochs), multi_compute_count))

        gs.is_interpolated[:] = ~np.frombuffer(lz4.decompress(response.has_data_buf), dtype=bool)

        # If there are counts in the metrics buffer... then we should be using a BatchedSeries.
        # This is an invalid response for converting to a GroupedSeries.
        assert not response.counts_buf

        # Since we have multiple computes, we will have multiple grouped series
        multi_compute_list: list[GroupedSeries] = []
        for compute_index in range(multi_compute_count):
            gs_copy = gs.copy()
            # Map a view from the values into the new gs_copy
            gs_copy.values = values_reshaped[:, :, compute_index]
            # Lets make sure the shape is correct, and that we have the right number of groups and epochs
            assert gs_copy.values.shape == (len(groups), len(epochs))
            multi_compute_list.append(gs_copy)

        return multi_compute_list


def _gs_to_df(context: SessionContext, gs: GroupedSeries) -> DataFrame:
    table_dict: dict[str, Any]
    if len(gs.groups) == 0:
        if len(gs.values.ravel()) > 0:
            raise ValueError(f"Unexpected - GroupedSeries with zero groups but values shape {gs.values.shape}")
        table_dict = {
            "epoch": np.array([], dtype=np.int64),
            "value": np.array([], dtype=np.float64),
            "is_interpolated": np.array([], dtype=bool),
        }
        main_table = context.from_arrow(pa.Table.from_pydict(table_dict))
        final_table = star_select(main_table, "epoch", to_timestamp_seconds(col("epoch")))
        return final_table

    table_dict = {"key_" + split_tag(tag)[0]: [] for tag in gs.groups[0]}

    for group in gs.groups:
        last_key = None
        for tag in sorted(list(group)):
            key, val = split_tag(tag)
            if "\t" in val:
                raise ValueError("tag values cannot contain tabs")
            if key == last_key:
                table_dict["key_" + key][-1] = f"{table_dict['key_' + key][-1]}\t{val}"
            else:
                table_dict["key_" + key].append(val)
            last_key = key

    for key, column in table_dict.items():
        if len(column) != len(gs.groups):
            raise ValueError(f"group with missing tag for key={key} in gsr")

    table_dict["value"] = pa.FixedSizeListArray.from_arrays(gs.values.ravel(), len(gs.epochs))
    main_table = context.from_arrow(pa.Table.from_pydict(table_dict))

    epoch_table = context.from_arrow(
        pa.Table.from_pydict(
            {
                "epoch": pa.FixedSizeListArray.from_arrays(gs.epochs, len(gs.epochs)),
                "is_interpolated": pa.FixedSizeListArray.from_arrays(gs.is_interpolated, len(gs.epochs)),
            }
        )
    )

    nested_table = main_table.join_on(epoch_table, lit(True))
    unnested_table = nested_table.unnest_columns("epoch", "is_interpolated", "value")
    # TODO(regis) For Rust testing we should not do the start_select. It takes a long time and it's unnecessary
    # return unnested_table
    final_table = star_select(unnested_table, "epoch", to_timestamp_seconds(col("epoch")))
    return final_table


def _df_to_gs(table: DataFrame) -> GroupedSeries:
    table = star_select(table, "epoch", to_unixtime(col("epoch")))
    has_interpolation = "is_interpolated" in table.schema().names
    aggs = [
        array_agg(col("epoch"), order_by=[col("epoch")]).alias("epoch"),
        array_agg(col("value"), order_by=[col("epoch")]).alias("value"),
    ]
    if has_interpolation:
        aggs.append(array_agg(col("is_interpolated"), order_by=[col("epoch")]).alias("is_interpolated"))
    array_table = table.aggregate(group_by=[col(name) for name in key_names(table)], aggs=aggs).filter(
        col("epoch").is_not_null()
    )
    total_value_df = None
    total_interp_df = None
    groups: list[tuple] = []
    for batch in array_table.execute_stream():
        arrow_batch: Any = batch.to_pyarrow()
        epoch_list: np.ndarray = arrow_batch.column("epoch").flatten().to_numpy()
        value_list: np.ndarray = arrow_batch.column("value").flatten().to_numpy()
        interp_list: np.ndarray | None = None
        if has_interpolation:
            # to_numpy is zero copy while to_pandas allows more flexibility
            interp_list = arrow_batch.column("is_interpolated").flatten().to_pandas().to_numpy()
        current_start = 0
        for i in range(arrow_batch.num_rows):
            current_end = current_start + len(arrow_batch.column("epoch")[i])
            value_df = pandas.DataFrame(
                data=value_list[current_start:current_end],
                columns=[f"{len(groups)}"],
                index=epoch_list[current_start:current_end],
            )
            interp_df = None
            if has_interpolation:
                if interp_list is None:
                    raise AssertionError("interpolated data missing when expected to be present")
                interp_df = pandas.DataFrame(
                    data=interp_list[current_start:current_end],
                    columns=[f"{len(groups)}"],
                    index=epoch_list[current_start:current_end],
                )
            new_group = tuple(
                sorted(
                    f"{name[4:]}:{val}"
                    for name in arrow_batch.column_names
                    if name.startswith("key_")
                    for val in str(arrow_batch.column(name)[i]).split("\t")
                )
            )
            groups.append(new_group)
            if total_value_df is None:
                total_value_df = value_df
            else:
                # This probably isn't very efficient but it's an easy way to join
                # on the epochs. If the sets of epochs for each group are the same we can
                # likely do much better.
                total_value_df = total_value_df.join(value_df, how="outer")

            if has_interpolation:
                if total_interp_df is None:
                    total_interp_df = interp_df
                else:
                    total_interp_df = total_interp_df.join(interp_df, how="outer")
            current_start = current_end

    if total_value_df is None:
        return GroupedSeries.empty()

    epochs = total_value_df.index.values

    is_interpolated = (
        np.all(total_interp_df.fillna(True).values, axis=1)
        if total_interp_df is not None
        else np.full((len(epochs),), False)
    )

    return GroupedSeries(
        epochs=epochs,
        values=total_value_df.values.T,
        groups=groups,
        is_interpolated=is_interpolated,
    )
