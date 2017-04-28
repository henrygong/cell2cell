class MappingModel(object):
    def fit(reference_data_array):
        raise NotImplementedError('fit method not implemented')

    def transform(query_data_array):
        raise NotImplementedError('transform method not implemented')

    def residuals(query_data_array):
        raise NotImplementedError('residuals method not implemented')


class PCAMappingModel(MappingModel):
    pass
