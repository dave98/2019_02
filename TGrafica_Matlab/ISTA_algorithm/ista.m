function x = ista(h, img_degraded, initialization, option, hfun)
  fprintf('Ejecutando ISTA\n');

  A = h;
  b = img_degraded;
  AtA = A'*A;
  evs = eig(AtA); %Obteniendo Eigen constants
  L = max(evs);
  l = 1/L;

  lambda = option.lambda;
  max_iteracciones = option.maxiter;
  tolerancia = option.tol;
  visualizar = option.vis;

  objk = func(initialization, b, A, lambda);
  xk = initialization

  fprintf('%6s %9s %9s\n','iter','f','sparsity');
  fprintf('%6i %9.2e %9.2e\n',0,objk,nnz(xk)/numel(xk));

  for i = 1:max_iteracciones
    x_old = xk;

    img_degraded = xk - l*(AtA*xk-A'*b);
    xk = (abs(img_degraded) - lambda/L).* sign(img_degraded)
    %xk = abs(img_degraded) - lambda/L;
    %xk = max(xk, 0)
    %xk = xk.*sign(img_degraded)

    if visualizar > 0
      fprintf('%6i %9.2e %9.2e\n',i,func(xk,b,A,lambda),nnz(xk)/numel(xk));
    end
    if norm(xk-x_old)/norm(x_old) < tolerancia
      fprintf('%6i %9.2e %9.2e\n',i,func(xk,b,A,lambda),nnz(xk)/numel(xk));
      fprintf('  converged at %dth iterations\n',i);
      break;
    end
  end
  x = xk;
end

function objk = func(xk, b, A, lambda)
  e = b - A*xk;
  objk = 0.5*(e)'*e + lambda*sum(abs(xk));
end
