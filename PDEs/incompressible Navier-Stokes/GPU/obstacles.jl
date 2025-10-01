# Hindernis
struct Rectangle
    i0::Int
    i1::Int
    j0::Int
    j1::Int
    function Rectangle(center, x_len, y_len, dx, dy)
        i0 = Int(round((center[1] - (x_len/2)) / dx))
        i1 = Int(round((center[1] + (x_len/2)) / dx))
        j0 = Int(round((center[2] - (y_len/2)) / dy))
        j1 = Int(round((center[2] + (y_len/2)) / dy))
        
        new(i0, i1, j0, j1)
    end
end

get_indices(rectangle) = rectangle.i0, rectangle.i1, rectangle.j0, rectangle.j1

function position_in_obstacle(obstacles, i, j)
    for obstacle in obstacles
        i0, i1, j0, j1 = get_indices(obstacle)
        if (i0 <= i <= i1 && j0 <= j <= j1)
            return true
        end
    end
    return false
end

function get_obstacle_indices(obstacles)
    obstacle_indices = Vector{Int}(undef, length(obstacles)*4)
    for (k, obstacle) in enumerate(obstacles)
        left = (k-1)*4 + 1
        right = (k-1)*4 + 4
        obstacle_indices[left:right] .= get_indices(obstacle)
    end
    return obstacle_indices
end