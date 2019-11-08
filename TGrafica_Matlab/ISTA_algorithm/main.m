%Basado en la implementación presentada por Seunghwan Yoo
clear; close all;
addpath(genpath('.'));

param.blurring = 1 % 1 para kernel gaussiano
param.obs = 1 % 1 para imagen seleccionada, 2 generación aleatoria

%-------------------------------CARGANDO IMAGEN--------------------------------
if param.obs == 1
  initial_image = im2double(imread('cat_1.jpg'));
  if ndims(initial_image) > 1
    initial_image = rgb2gray(initial_image);
  end
  sized_image = initial_image(51:150, 51:150); %Seleccionando un pedazo de la imagen

else
  sparsity = 0.01;
  %sized_image = obj_sparse(sparsity, 20, 20) %Construyendo imagen de forma aleatoria
  load sized_image.mat;
end

x_val = sized_image(:); %Vectorizando

%------------------------------DITORSIONANDO--------------------------------
distortionkernel = fspecial('gaussian', [22,22], 1.2); %Tamao de kernel e indice de distorsion
laplaciankernel = [0 0.25 0; 0.25 -1 0.25; 0 0.25 0]; % Laplacian

%Creando matriz lexicográfica (no python :'v)
[h, h_2d] = lexicoH(sized_image, distortionkernel);
[c, c_2d] = lexicoH(sized_image, laplaciankernel);

%Degradando
y=h*x_val; % Blurred vector
y_2d = reshape(y, size(sized_image)); % Blurred 2D, solo para mostrar

%Mostrando
figure('Position',[0,100,450,450]),imagesc(sized_image);title('Segmento Original');caxis([0,1]);
figure('Position',[450,100,450,450]),imagesc(y_2d);title('Segmento Degradado');caxis([0,1]);

% ---------------------------------------ISTA configuration---------------------------------+
opt.tol = 10^(-6); %Criterio de Parada
opt.maxiter = 200; %Maximo de Iteracciones
opt.lambda = 0.01; %Lipschitz
opt.vis = 0; % Imprimir proceso -> 0:nada, 1:log, 2:log+figure

x_initial = y;

%-----------------------------------------ISTA algorithm--------------------------------------
fprintf('LASSO with ISTA\n');
[x_ista_i] = ista(h, y, x_initial,opt,0);
figure('Position',[900,100,450,450]),imagesc(reshape(x_ista_i,size(sized_image)));title('ISTA reconstruction');caxis([0,1]);
