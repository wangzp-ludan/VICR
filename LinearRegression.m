function [a,b,r]=LinearRegression_multiple(y,x);
[Number_pixels,Number_bands,Number_images]=size(x);

%%%%%linear regression modeling, suitable for any c1 (=1 or >1)
for i=1:Number_bands
    for j=1:Number_images
        xx(:,j)=x(:,i,j);
    end
    yy=y(:,i);
    X=[xx,ones(Number_pixels,1)];
    [coe,~,residual,~,~] = regress(yy,X);
    %[XL,YL,XS,YS,beta,PCTVAR,MSE,stats] = plsregress(xx,yy);
    a(:,i)=coe(1:Number_images);
    b(i)=coe(Number_images+1);
    r(:,i)=residual;
end
