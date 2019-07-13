X = 1;
Y = 0.2;
k = 1.3;
ro = 2400;
cp = 700;
alfa = k/(ro*cp);

nx = 41;
ny = 11;

dx = X/(nx-1);
dy = Y/(ny-1);

dt = 1;
tson = 3600;
tilk = 0;
time_step = tson/dt+1;

tx = alfa*dt/dx^2;
ty = alfa*dt/dy^2;
h = 1000;
Ts = 1223;

A1 = zeros (nx*ny,nx*ny);
A2 = zeros (nx*ny,nx*ny);
G  = zeros (nx*ny,1);
T  = 20*ones(nx*ny,1);

for i = 1:nx
    G(i) = tx*h*dx*Ts/k;
end

for i = 1
    for j = 1
        A1 ((i-1)*nx+j,(i-1)*nx+j) = 1+tx+ty+tx*h*dx/2/k;
        A1 ((i-1)*nx+j,(i-1)*nx+j+1) = -tx;
        A1 ((i-1)*nx+j,i*nx+j) = -ty;
        
        A2 ((i-1)*nx+j,(i-1)*nx+j) = 1-tx-ty-tx*h*dx/2/k;
        A2 ((i-1)*nx+j,(i-1)*nx+j+1) = tx;
        A2 ((i-1)*nx+j,i*nx+j) = ty;
        
    end
    
    for j = 2:nx-1
        A1 ((i-1)*nx+j,(i-1)*nx+j) = 1+tx+ty+tx*h*dx/2/k;
        A1 ((i-1)*nx+j,(i-1)*nx+j-1) = -tx/2;
        A1 ((i-1)*nx+j,(i-1)*nx+j+1) = -tx/2;
        A1 ((i-1)*nx+j,i*nx+j) = -ty;
        
        A2 ((i-1)*nx+j,(i-1)*nx+j) = 1-tx-ty-tx*h*dx/2/k;
        A2 ((i-1)*nx+j,(i-1)*nx+j-1) = tx/2;
        A2 ((i-1)*nx+j,(i-1)*nx+j+1) = tx/2;
        A2 ((i-1)*nx+j,i*nx+j) = ty;
    end
    
    for j =nx
        A1 ((i-1)*nx+j,(i-1)*nx+j) = tx+ty+tx*h*dx/2/k+1;
        A1 ((i-1)*nx+j,(i-1)*nx+j-1) = -tx;
        A1 ((i-1)*nx+j,i*nx+j) = -ty;
        
        A2 ((i-1)*nx+j,(i-1)*nx+j) = 1-tx-ty-tx*h*dx/2/k;
        A2 ((i-1)*nx+j,(i-1)*nx+j-1) = tx;
        A2 ((i-1)*nx+j,i*nx+j) = ty;
    end    
end

for i = 2:ny-1
    for j = 1
        A1 ((i-1)*nx+j,(i-1)*nx+j) = tx+ty+1;
        A1 ((i-1)*nx+j,(i-1)*nx+j+1) = -tx;
        A1 ((i-1)*nx+j,(i-2)*nx+j) = -ty/2;
        A1 ((i-1)*nx+j,i*nx+j) = -ty/2;
        
        A2 ((i-1)*nx+j,(i-1)*nx+j) = 1-tx-ty;
        A2 ((i-1)*nx+j,(i-1)*nx+j+1) = tx;
        A2 ((i-1)*nx+j,(i-2)*nx+j) = ty/2;
        A2 ((i-1)*nx+j,i*nx+j) = ty/2;
    end
    
    for j = 2:nx-1
        A1 ((i-1)*nx+j,(i-1)*nx+j) = tx+ty+1;
        A1 ((i-1)*nx+j,(i-1)*nx+j-1) = -tx/2;
        A1 ((i-1)*nx+j,(i-1)*nx+j+1) = -tx/2;
        A1 ((i-1)*nx+j,(i-2)*nx+j) = -ty/2;
        A1 ((i-1)*nx+j,i*nx+j) = -ty/2;
        
        A2 ((i-1)*nx+j,(i-1)*nx+j) = 1-tx-ty;
        A2 ((i-1)*nx+j,(i-1)*nx+j-1) = tx/2;
        A2 ((i-1)*nx+j,(i-1)*nx+j+1) = tx/2;
        A2 ((i-1)*nx+j,(i-2)*nx+j) = ty/2;
        A2 ((i-1)*nx+j,i*nx+j) = ty/2;
    end
    
    for j = nx
        A1 ((i-1)*nx+j,(i-1)*nx+j) = tx+ty+1;
        A1 ((i-1)*nx+j,(i-1)*nx+j-1) = -tx;        
        A1 ((i-1)*nx+j,(i-2)*nx+j) = -ty/2;
        A1 ((i-1)*nx+j,i*nx+j) = -ty/2;
        
        A2 ((i-1)*nx+j,(i-1)*nx+j) = 1-tx-ty;
        A2 ((i-1)*nx+j,(i-1)*nx+j-1) = tx;
        A2 ((i-1)*nx+j,(i-2)*nx+j) = ty/2;
        A2 ((i-1)*nx+j,i*nx+j) = ty/2;
    end
end

for i = ny
    for j = 1
        A1 ((i-1)*nx+j,(i-1)*nx+j) = tx+ty+1;
        A1 ((i-1)*nx+j,(i-1)*nx+j+1) = -tx;
        A1 ((i-1)*nx+j,(i-2)*nx+j) = -ty;
        
        A2 ((i-1)*nx+j,(i-1)*nx+j) = 1-tx-ty;
        A2 ((i-1)*nx+j,(i-1)*nx+j+1) = tx;
        A2 ((i-1)*nx+j,(i-2)*nx+j) = ty;
    end
    
    for j = 2:nx-1
        A1 ((i-1)*nx+j,(i-1)*nx+j) = tx+ty+1;
        A1 ((i-1)*nx+j,(i-1)*nx+j-1) = -tx/2;
        A1 ((i-1)*nx+j,(i-1)*nx+j+1) = -tx/2;
        A1 ((i-1)*nx+j,(i-2)*nx+j) = -ty;
        
        A2 ((i-1)*nx+j,(i-1)*nx+j) = 1-tx-ty;
        A2 ((i-1)*nx+j,(i-1)*nx+j-1) = tx/2;
        A2 ((i-1)*nx+j,(i-1)*nx+j+1) = tx/2;
        A2 ((i-1)*nx+j,(i-2)*nx+j) = ty;
    end
    
    for j = nx
        A1 ((i-1)*nx+j,(i-1)*nx+j) = tx+ty+1;
        A1 ((i-1)*nx+j,(i-1)*nx+j-1) = -tx;
        A1 ((i-1)*nx+j,(i-2)*nx+j) = -ty;
        
        A2 ((i-1)*nx+j,(i-1)*nx+j) = 1-tx-ty;
        A2 ((i-1)*nx+j,(i-1)*nx+j-1) = tx;
        A2 ((i-1)*nx+j,(i-2)*nx+j) = ty;
    end
end

for e = 1:time_step-1
T(:,:,e+1) = A1\(A2*T(:,:,e)+G);
end
C = reshape (T(:,:,time_step),nx,ny);
D = C';