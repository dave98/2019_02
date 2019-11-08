function [imgkernel_2d] = c_h2d(img_2d, kernel_2d)
  % Aplica Kernel 2d en toda la Imagen
  [ix, iy] = size(img_2d); % Tamaño imagen
  h_temp = zeros(ix, iy);

  [kx, ky] = size(kernel_2d); % Tamaño Kernel
  h_temp(1:kx, 1:ky) = kernel_2d; % Estableciendo kernel esquina superior izquierda
  imgkernel_2d = circshift(h_temp, [-floor(kx/2),-floor(ky/2)]); % Centrando Kernel
end
