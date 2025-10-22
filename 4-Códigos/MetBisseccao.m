function bissecao(f, a, b, TOL, N)
%f=inline('x*x-4');
i = 1;
  fa = f(a); 
  fb = f(b);
  if fa*fb>0
     disp("Erro: A função tem mesmo sinal nos pontos a e b");
  else
      disp("Int\t a\t   b \t  solução  f(x)\t   tolerância")
     for i=1:N
     #iteração da bisseção
     r=(a+b)/2;
     err=abs((b-a)/2);
     fr= f(r);
    if fr==0
        fprintf("Solução exata x=%0.6f foi encontrada\n",r);
        break
    endif
    if err<TOL
        fprintf("Tolerância atingida.");
        break
    endif
fprintf("%3i %.5f %.5f %.5f %.5f %.6f\n",i,a,b,r,fr,err);
    i = i+1;
    #bissecção troca de intervalo
    if (fa * fr > 0)
      a = r;
      fa = fr;
    else
      b = r;
    endif
  endfor
  endif
  fprintf("Solução: %f",r)
  endfunction