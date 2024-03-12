function [phi_2] = two_order(phi,h)
% 离散序列二阶导数
Phase_length = length(phi); 
phi_2 = zeros(1,Phase_length);
for n = 4:Phase_length-3
    phi_2(n) = 4*phi(n) + (phi(n+1)+phi(n-1)) - 2*(phi(n+2)+phi(n-2)) -(phi(n+3)+phi(n-3));
    phi_2(n) = phi_2(n) / (16*h*h);
end
end

