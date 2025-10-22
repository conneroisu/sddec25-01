function x = Gauss(A, b)
% Implementação do método de eliminação de Gauss para resolver sistemas
% lineares Ax = b.

    % Verificação de pré-condição
    [m, n] = size(A);
    if m ~= n
        error('A matriz deve ser quadrada');
    end    
    % Adicionando a coluna b à matriz A
    Aug = [A, b];    
    % Inicializando vetor de solução x
    x = zeros(n, 1);    
    % Eliminação gaussiana
    for k = 1:n-1
        [~, i] = max(abs(Aug(k:n, k)));
        ipr = i + k - 1;
        if ipr ~= k
            Aug([k, ipr], :) = Aug([ipr, k], :);
        end
        for i = k+1:n
            pivor = Aug(i, k) / Aug(k, k);
            Aug(i, k:n+1) = Aug(i, k:n+1) - pivor*Aug(k, k:n+1);
        end
    end    
    % Retrosubstituição
    x(n) = Aug(n, n+1) / Aug(n, n);
    for i = n-1:-1:1
        x(i) = (Aug(i, n+1) - Aug(i, i+1:n)*x(i+1:n)) / Aug(i, i);
    end
end