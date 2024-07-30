#                                           Álgebra Linear Aplicada
#
# Nome: Emanuel Oliveira Souza
# Nº Matrícula: 12991371
# Data: Atualizado em 19/07/2024
#
# Descrição do Trabalho:
#     Implementação do algoritmo de conversão de uma matriz para a forma de Hessenberg e o algoritmo QR
#     e QR com mudança dupla da origem (Double Shift).
#
#

try
    using LinearAlgebra, BlockDiagonals, LinearSolve
catch
    using Pkg;
    dependencies = [
    "LinearAlgebra",
    "BlockDiagonals",
    "LinearSolve"
    ]
    Pkg.add(dependencies)

end

# Compute the specific vector v for the Householder reflection

function v_householder(x::Vector)
    
    n = length(x)
    x = x / norm(x)
    s = x[2:n]' * x[2:n]
    v = [1; x[2:n]]
    
    if s == 0
        β = 0
    else
        μ = sqrt(x[1]^2 + s)
        if x[1] <= 0
            v[1] = x[1] + μ
        else
            v[1] = -s / (x[1] + μ)
        end
        β = 2 * v[1]^2 /(s + v[1]^2)
        v = v / v[1]
    end

    return v, β

end

# Compute the upper Hessenberg matrix similar to matrix A.

function hessenberg_householder(A::Matrix)

    n = maximum(size(A))
    Q = Matrix{Float64}(I, n, n)
    H = A

    for i in 1:n - 2
        # Create the vector v from the Householder transformation
        
        v, β = v_householder(H[i + 1:n, i])
        R = Matrix{Float64}(I, length(v), length(v)) - β * v * v'

        # Multiplication by the matrix resulting from the Householder transformation - P * H * P
        
        P = BlockDiagonal([Matrix{Float64}(I, i, i), R])
        H[i + 1:n, i:n] = R * H[i + 1:n, i:n]
        H[1:n, i + 1:n] = H[1:n,i + 1:n] * R
        
        # Save successive multiplications P_1 * P_2 * ... * P_{n-2}
        
        Q = Q * P
    end

    return H
end

# Creates the Givens rotation matrix in the plane (iK)

function G_rotation(x_i::Float64, x_k::Float64, k::Int64, i::Int64, n::Int64)

    G = Matrix{Float64}(I, n, n)
    r = sqrt(x_i^2 + x_k^2)
    c = x_i / r
    s = x_k / r

    G[i, i] = c
    G[i, k] = s
    G[k, i] = -s
    G[k, k] = c

    return G
end

# Determine the QR factorization of the upper Hessenberg matrix H using Givens rotations.

function qr_givens(H::Matrix)

    n = maximum(size(H))
    Q = Matrix{Float64}(I, n, n)
    R = H

    for k = 1:n - 1

        g = G_rotation(R[k, k], R[k + 1, k], k + 1, k, n)
        R = g * R
        Q = g * Q
    end

    return Q', R
end

# Uses the QR method to estimate the eigenvalues of a matrix in higher Hessenberg form. This function returns the triangular matrix
# upper T and eigenvalues.

function eigenvalue_qr(H::Matrix, n_it::Int64, tol::Float64 = 1e-10)
    T = H
    for i in 1:n_it
        Q, R = qr_givens(T)
        T = R * Q
        T_ = copy(T)
        T_[abs.(T) .< tol] .= 0.0
        if istriu(T_)
            return diag(T), T
        end
    end
    println("Maximum iteration reached")
    
    return diag(T), T
end

# Finds the eigenvalues of a matrix in higher Hessenberg form using the QR method with double shift.
# Note: This implementation was based on the book Numerical Mathematics.

function eigenvalue_qr_double_shift(H::Matrix, n_int::Int64, tol::Float64 = 1e-10)

    T = complex(copy(H))
    n = size(T)[1]
    it = 0

    for k = n:-1:2

        I_k = Matrix{Float64}(I, k, k)
        while abs(T[k, k -1]) > tol * (abs(T[k, k]) + abs(T[k - 1, k - 1]))
            it += 1
            if it > n_int
                return diag(T), T
            end
            μ = T[k, k]
            Q, R = qr(T[1:k, 1:k] - μ * I_k)
            T[1:k, 1:k] = R * Q + μ * I_k
            if k > 2
                trace = abs(T[k - 1, k - 1]) + abs(T[k - 2, k - 2])
                if abs(T[k - 1, k - 2]) <= tol * trace
                    eig = eigvals(T[k - 1:k, k - 1:k])
                    Q, R = qr(T[1:k, 1:k] - eig[1] * I_k)
                    T[1:k, 1:k] = R * Q + eig[1] * I_k
                    Q, R = qr(T[1:k, 1:k] - eig[2] * I_k)
                    T[1:k, 1:k] = R * Q + eig[2] * I_k
                end
            end
        end
        T[k, k - 1] = 0.0
    end
    I_2 = Matrix{Float64}(I, 2, 2)
    while (abs(T[2, 1]) < tol * (abs(T[2, 2]) + abs(T[1, 1]))) & (it <= n_int)
        it += 1
        μ = T[2, 2]
        Q, R = qr(T[1:2, 1:2] - μ * I_2)
        T[1:2, 1:2] = R * Q + μ * I_2
    end

    return diag(T), T
end

# Returns the eigenvectors associated with a set of eigenvalues using the inverse power method

function inverse_power(A::Matrix, eigen_values::Vector, n_it::Int64)

    n = size(A)[1]
    x_0 = ones(n)
    I_n = Matrix{Float64}(I, n, n)
    eigen_vec = zeros(n, n)
    
    if eltype(eigen_values) == ComplexF64
        eigen_vec = complex(eigen_vec)
        I_n = complex(I_n)
        x_0 = complex(x_0)
        A = complex(A)
    end
    
    for (i, eig) in enumerate(eigen_values)
        σ = eig - 0.00001
        for k in 1:n_it
            prob = LinearProblem(A - σ * I_n, x_0)
            sol = solve(prob)
            x_k = sol.u
            x_0 = x_k / norm(x_k)
        end
        eigen_vec[:,i] = x_0
    end

    return eigen_vec
end

# Calculates the error between the calculated eigenvalues and eigenvectors.

function error(A::Matrix, eigen_values::Vector, eigen_vect::Matrix)

    erro = []

    for (i, eig) in enumerate(eigen_values)
        vect = eigen_vect[:, i]
        e = norm(A * vect - eig * vect)
        erro = [erro; e]
    end
    
    return erro
end



function testar()
    println("Exemplo 1 - Matriz com autovalores reais e distintos")
    A_1 = [1.0 2.0 3.0 4.;2.0 6.0 7.0 8.0;3.0 7.0 11.0 12.0;4.0 8.0 12.0 16.0];
    display(A_1);

    println("Matriz na forma de Hessenberg superior")
    H_1 = hessenberg_householder(A_1);
    display(H_1);

    println("Autovalores utilizando o método QR")
    eig_values_A_1, T_1 = eigenvalue_qr(H_1, 200);
    display(eig_values_A_1);

    println("Autovetores associados aos autovalores")
    eig_vect_A_1 = inverse_power(A_1, eig_values_A_1, 100);
    display(eig_vect_A_1);

    println("Erro entre as aproximações calculadas para os autovetores e autovalores")
    erro_1 = error(A_1, eig_values_A_1, eig_vect_A_1);
    display(erro_1);

    println("================================================")

    println("Exemplo 2 - Matrix real com dois pares de autovalores complexos")
    A_2 = [1.0 -1.0 0.0 0.0; 1.0 1.0 0.0 0.0;0.0 0.0 0.0 -1.0;0.0 0.0 1.0 1.0];
    display(A_2);

    println("Matriz resultante utilizando o método QR - não converge")
    eeig_values_A_2, T_2 = eigenvalue_qr(A_2, 500);
    display(T_2);

    println("Para solucionar isto usaremos o método QR com double shift")
    println("Autovalores associados:")
    eig_values_A_2, T_2 = eigenvalue_qr_double_shift(A_2, 200);
    display(eig_values_A_2);

    println("Autovetores associados aos autovalores")
    eig_vect_A_2 = inverse_power(A_2, eig_values_A_2, 250);
    display(eig_vect_A_2);

    println("Erro entre as aproximações calculadas para os autovetores e autovalores")
    erro_2 = error(A_2, eig_values_A_2, eig_vect_A_2);
    display(erro_2);

    println("================================================")

    println("Exemplo 3 - Matriz simétrica")
    A_3 = [1.0 2.0 3.0 4.0;2.0 5.0 6.0 7.0;3.0 6.0 8.0 9.0;4.0 7.0 9.0 10.0]
    display(A_3);

    println("Matriz na forma de Hessenberg superior")
    H_3 = hessenberg_householder(A_3);
    display(H_3);

    println("Autovalores utilizando o método QR")
    eig_values_A_3, T_3 = eigenvalue_qr(H_3, 500);
    display(eig_values_A_3);

    println("Autovetores associados aos autovalores")
    eig_vect_A_3 = inverse_power(A_3, eig_values_A_3, 500);
    display(eig_vect_A_3)

    println("Erro entre as aproximações calculadas para os autovetores e autovalores")
    erro_3 = error(A_3, eig_values_A_3, eig_vect_A_3);
    display(erro_3);

    println("================================================")

    println("Exemplo 4 - Matriz vista em sala")
    A_4 = [0.0 0.0 1.0;1.0 0.0 0.0;0.0 1.0 0.0];
    display(A_4);

    println("Matriz resultando o utilizando o método QR")
    eig_values_A_4, T_4 = eigenvalue_qr(A_4, 200);
    display(T_4);

    println("Matriz resultando o utilizando o método QR com double shift")
    eig_values_A_4, T_4 = eigenvalue_qr(A_4, 200);
    display(T_4);

    println("Mesmo com double shift, ambos os métodos implementados não convergem para esta estrutura de matriz")

end

testar()


