function [ Xc, Yc ] = Ellipse_Fitting(xs,ys)

X = [xs.^2;
    xs.*ys;
    ys.^2;
    xs;
    ys;
    ones(1,length(xs))];
H = zeros(6);
H(1,3)=2;
H(3,1)=2;
H(2,2)=-1;
S = X*X';

% %% 算法一
% [V,L] = eig(S,H);
% L = diag(L);
% 
% for i=1:6
%     if L(i)<=0
%         continue;
%     end
%     
%     W = V(:,i);
%     
% 	if W'*H*W<0
%         continue;
%     end
%     
%     W = sqrt(1/(W'*H*W))*W;
% 
%     A = W(1); B = W(2); C = W(3); D = W(4); E = W(5); F = W(6); 
%     funs = @(x,y) A*x.^2 + B*x.*y + C*y.^2 + D*x + E*y + F; 
%     figure; 
%     hold on; 
%     scatter(xs,ys,[],'.'); 
%     fimplicit(funs)
% 
% 	Xc = (B*E-2*C*D)/(4*A*C-B^2);
% 	Yc = (B*D-2*A*E)/(4*A*C-B^2);
% end

%% 算法二
[V,L] = eig(S);
EE = zeros(1,6);
for i=1:size(V,2)
    EE(i) = V(:,i)'*S*V(:,i);
end
[~,I] = min(EE);
W = V(:,I);

% [~, index] = sort(EE, 'ascend');
% I = 1;
% while ( V(1,index(I))*V(3,index(I)) < 0 )
%     I = I + 1;
%     if I > fsize(V,2)
%         I = 1;
%         break;
%     end
% end
% W = V(:,index(I));

A = W(1); B = W(2); C = W(3); D = W(4); E = W(5); F = W(6); 
funs = @(x,y) A*x.^2 + B*x.*y + C*y.^2 + D*x + E*y + F; 
figure; 
hold on; 
scatter(xs,ys,[],'.'); 
fimplicit(funs,'LineWidth',2,'Color','r')
xlabel('I');ylabel('Q');

Xc = (B*E-2*C*D)/(4*A*C-B^2);
Yc = (B*D-2*A*E)/(4*A*C-B^2);

scatter(Xc,Yc,30,'r','filled'); 

end