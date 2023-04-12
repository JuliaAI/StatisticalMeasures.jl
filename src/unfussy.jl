unfussy(measure) = measure
unfussy(measure::API.FussyMeasure) = getfield(measure, :atom)
