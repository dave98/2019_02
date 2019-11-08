function [H, applied_kernel_2d] = lexicoH(img_2d, kernel_2d)
  % Taken from Seunghwan Implementation

  [ix, iy] = size(img_2d);
  applied_kernel_2d = c_h2d(img_2d, kernel_2d);
  h_row = zeros(ix, ix*iy);

  for i=0:ix-1
    applied_kernel_2d_shift = circshift(applied_kernel_2d, [i, 0]);
    h_row(i+1,:) = applied_kernel_2d_shift(:)';
  end
  for j=0:iy-1
    h_row_shift = circshift(h_row,[0, j*ix]);
    H(j*ix+1:(j+1)*ix,:) = h_row_shift;
  end
end
