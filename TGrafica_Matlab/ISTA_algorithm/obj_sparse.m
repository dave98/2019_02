function answer = obj_sparse(ratio, x, y)
  N = x*y;
  answer = zeros(N, 1);
  S = N*ratio;
  rseq = randperm(N);
  answer(rseq(1:S)) = rand([1, S]);
  answer = reshape(answer, x, y)
end
