function is_uppercase(char::Char)
    i = Int(char)
    i > 64 && i < 91
end

"""
    snakecase(str, del='_')

Return the snake case version of the abstract string or symbol, `str`, as in

    snakecase("TheLASERBeam") == "the_laser_beam"

"""
function snakecase(str::AbstractString; delim='_')
    snake = Char[]
    n = length(str)
    for i in eachindex(str)
        char = str[i]
        if is_uppercase(char)
            if i != 1 && i < n &&
                !(is_uppercase(str[i + 1]) && is_uppercase(str[i - 1]))
                push!(snake, delim)
            end
            push!(snake, lowercase(char))
        else
            push!(snake, char)
        end
    end
    return join(snake)
end

snakecase(s::Symbol) = Symbol(snakecase(string(s)))

"""
    check_pools(A::UnivariateFiniteArray, B::CategoricalArrays.CatArrOrSub)

*Private method.*

Check that the class pool of `A` coincides with the class pool of `B`, as sets. If both
`A` and `B` are ordered, check the pools have the same ordering.

If a check fails, throw an exception, and otherwise return `nothing`.

"""
function API.check_pools(
    A::UnivariateFiniteArray,
    B::CategoricalArrays.CatArrOrSub,
    )

    first_nonmissing_index = findfirst(x->!ismissing(x), A)
    element_of_A = A[first_nonmissing_index]
    classes_a = CategoricalArrays.levels(element_of_A)
    classes_b = CategoricalArrays.levels(B)
    if  CategoricalArrays.isordered(A) && CategoricalArrays.isordered(B)
        classes_a == classes_b || throw(API.ERR_POOL_ORDER)
    else
        Set(classes_a) == Set(classes_b) || throw(API.ERR_POOL)
    end
    return nothing
end

