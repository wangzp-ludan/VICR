clear
clc

path='Data/';
outpath='Predict/';
Files_L=dir([path,'*LC08.tif']);
cloud=geotiffread([path,'mask.tif']);

% date time
for i=1:length(Files_L)
    name=Files_L(i).name;
    x=2;
    yy=str2num(name(x+1:x+4));
    mm=str2num(name(x+5:x+6));
    dd=str2num(name(x+7:x+8));
    doy(i)=datenum(yy,mm,dd);
end

%aim image
aim=7;
doy=doy-doy(aim);

[rank_i,id]=sort(abs(doy));


iii=length(id);
% similar pixel
similar_num=20;
% windows: 2*w+1
w=15;
%% run code

n=0;
for i=1:6
    mask1(:,:,i)=cloud;
end
for i=1:iii
    [l,ref]=geotiffread([path,Files_L(id(i)).name]);
    info=geotiffinfo([path,Files_L(id(i)).name]);
    l(l<=0)=0;
    if i==1
        aim(1,2)=i;
        cloud_truth_value=l;
        l(mask1==1)=0;
        Landsat_predict=l*100;
    else
        n=n+1;
        Landsat_known(:,:,:,n)=l*100;
        mask_known(:,:,n)=(sum(l~=0,3)~=6);
    end
    if n==12
        break;
    end
end
[Number_row,Number_col,Number_bands,Number_images]=size(Landsat_known);

iii=n+1;
% linear interp nodata
for k=1:Number_bands
    for i=1:iii-1
        [row,col]=find(Landsat_known(:,:,k,i)==0 & cloud==1);
        for m=1:length(row)
            doy_x=doy(id(2:iii));
            doy_x1=doy_x(i);
            Landsat_y=Landsat_known(row(m),col(m),k,:);
            Landsat_y=Landsat_y(:);
            Landsat_y(isnan(Landsat_y))=0;
            doy_x(Landsat_y==0)=[];
            Landsat_y(Landsat_y==0)=[];
            if length(doy_x)>1
                interpl_y = interp1(doy_x,Landsat_y,doy_x1,'linear');
                interpl_y(interpl_y<0)=0;
                Landsat_known(row(m),col(m),k,i) = interpl_y;
            end
        end
    end
end


tic
%% Calculate

Landsat_result=zeros(size(Landsat_predict));
Landsat_predict_reshape=reshape(Landsat_predict,[Number_row*Number_col Number_bands]);


stats=regionprops(cloud==1,'Image','BoundingBox','PixelList');
j=1;
stats_len=length(stats);

while j<=stats_len
    disp(j);
    cloud_j=zeros(size(cloud));
    x_loc=stats(j).BoundingBox(1)+0.5;
    y_loc=stats(j).BoundingBox(2)+0.5;
    wid=stats(j).BoundingBox(3);
    hig=stats(j).BoundingBox(4);
    cloud_j(y_loc:y_loc+hig-1,x_loc:x_loc+wid-1)=stats(j).Image;
    cloud_in=logical(cloud_j);
    filter_cloud=ones(2*w+1);
    cloud_bufferj=imdilate(cloud_j,filter_cloud);
    cloud_bufferj(cloud==1 & cloud_in==1)=2;
    cloud_bufferj(cloud==1 & cloud_in==0)=0;
    fill_locj=stats(j).PixelList;
    
    mask=(cloud_bufferj==1);
    
    sRMSE=100;
    useful_images=0;
    k1=0;Z2_L1=zeros(size(Landsat_predict));
    a=0;
    
    while k1==0
        num=0;Z2_L=Z2_L1;A=a';
        if useful_images~=0
            known_useful_reshape=reshape(Landsat_known_useful,[Number_row*Number_col Number_bands useful_images-length(useless_num)]);
        end
        useless_num=[];
        useful_images=useful_images+1;
        Landsat_known_useful=Landsat_known(:,:,:,1:useful_images);
        mask_known_useful=mask_known(:,:,1:useful_images);
        for i=1:useful_images
            known_reshape=reshape(Landsat_known_useful(:,:,:,i),[Number_row*Number_col Number_bands]);
            cloud_known=mask_known_useful(:,:,i)==0;
            cloud_known(cloud_bufferj==0)=[];
            if sum(cloud_known) > 0.95*sum(sum(cloud_bufferj>=1))
                num=num+1;
                Landsat_known_reshape(:,:,num)=known_reshape;
            else
                useless_num=[useless_num,i];
            end
        end
        Landsat_known_useful(:,:,:,useless_num)=[];
        mask_known_useful(:,:,useless_num)=[];
        if num==0
            continue;
        end
        mask_known_useful=sum(mask_known_useful,3)>0;
        Landsat_known_reshape(mask==0 | mask_known_useful==1,:,:)=[];
        Landsat_aim_reshape=Landsat_predict_reshape;
        Landsat_aim_reshape(mask==0 | mask_known_useful==1,:)=[];
        
        [a,b,~]=LinearRegression_multiple(Landsat_aim_reshape,Landsat_known_reshape);
        Z1_L=zeros(size(Landsat_predict));
        clear Landsat_known_reshape
        
        
        for k=1:Number_bands
            for i=1:num
                Z1_L(:,:,k)=Z1_L(:,:,k)+a(i,k)*Landsat_known_useful(:,:,k,i);
            end
            Z2_Lk=Z1_L(:,:,k)+b(k);
            Z2_L1(:,:,k)=Z2_Lk;
        end
        
        cloud_bufferj_test=imdilate(cloud_bufferj>0,ones(4*w+1));
        cloud_bufferj_test=cloud_bufferj_test.*(cloud==0).*(cloud_bufferj==0).*(mask_known_useful==0);
        for m=1:6
            moni_near=Z2_L1(:,:,m);
            truek=Landsat_predict(:,:,m);
            xx=(moni_near>0 & truek>0 & cloud_bufferj_test==1);
            true_cloud_value=cloud_truth_value(:,:,m);
            vip_cloud_value=Z2_L1(:,:,m);
            
            yy=(cloud_bufferj==2 & true_cloud_value>0);
            vip_cloud_value(yy==0)=[];
            true_cloud_value(yy==0)=[];
            rmse_cloud(m,1)=sqrt(sum((vip_cloud_value/100-true_cloud_value).^2)/sum(sum(yy)));
            
            moni_near(xx==0)=[];
            truek(xx==0)=[];
            rmse(m,1)=sqrt(sum((truek-moni_near).^2)/sum(sum(xx)))/100;
        end
        if sRMSE<sum(rmse)
            k1=1;
        elseif num==12 || useful_images==iii-1
            k1=1;
            A=a';
            Z2_L=Z2_L1;
            known_useful_reshape=reshape(Landsat_known_useful,[Number_row*Number_col Number_bands useful_images-length(useless_num)]);
        end
        sRMSE=sum(rmse);
    end
    
    delta=reshape((Landsat_predict-Z2_L),[Number_row*Number_col Number_bands]);
    Z2_L_reshape=reshape(Z2_L,[Number_row*Number_col Number_bands]);
    
    
    [kmeans_row,kmeans_col]=find(cloud_bufferj>0);
    index=find(cloud_bufferj>0);
    preclass=Z2_L_reshape(index,:);
    
    maps=[kmeans_row,kmeans_col,index];
    
    nocloud_loc=find(cloud_bufferj==1);
    cloud_loc=find(cloud_bufferj==2);
    times=length(stats(j).PixelList(:,1));
    delta_result=zeros(times,Number_bands);
    %     parfor k=1:times
    for k=1:times
        aim_row=fill_locj(k,2);
        aim_col=fill_locj(k,1);
        
        aim_value=known_useful_reshape(cloud_loc(k),:,:);
        
        [index_insect,index_index,~]=intersect(maps(:,3),nocloud_loc);
        
        nocloud_series=known_useful_reshape(index_insect,:,:);
        series_differ=(nocloud_series-aim_value).^2;
        TWSD=0;
        if isempty(series_differ)
            delta_result(k,:)=0;
            continue;
        end
        for n=1:length(series_differ(1,1,:))
            TWSD=series_differ(:,:,n)*abs(A(:,n))+TWSD;
        end
        TWSD=sqrt(TWSD/Number_bands);
        
        [~,sort_index]=sort(TWSD);
        if length(sort_index)>similar_num
            sort_index=sort_index(1:similar_num);
        end
        D=maps(index_index(sort_index),1:2);
        D_differ=sqrt(sum((D-[aim_row,aim_col]).^2,2));
        S_differ=TWSD(sort_index);
        
        S_differ=(S_differ-min(S_differ))/(max(S_differ)-min(S_differ))+1;
        D_differ=(D_differ-min(D_differ))/(max(D_differ)-min(D_differ))+1;
        W_SD=1./(S_differ.*D_differ)./sum(1./(S_differ.*D_differ));
        delta_k=delta(index_insect(sort_index),:);
        if isempty(delta_k) | isnan(W_SD)
            delta_result(k,:)=0;
            continue;
        end
        delta_k1=sum(W_SD.*delta_k);
        delta_result(k,:)=delta_k1;
    end
    
    Landsat_result_b=zeros(size(Landsat_result(:,:,1)));
    for b=1:6
        Landsat_result_b(cloud_in)=Z2_L_reshape(cloud_in,b)+delta_result(:,b);
        delta_result_zero=delta_result(:,b)==0;
        Landsat_result_b(cloud_in(delta_result_zero))=0;
        Landsat_result(:,:,b)=Landsat_result(:,:,b)+Landsat_result_b;
    end
    j=j+1;
end
Landsat_predict=Landsat_predict+single(Landsat_result);
export_name=['VICR_',Files_L(aim(1)).name];
Landsat_predict=Landsat_predict/100;
geotiffwrite([outpath,export_name],Landsat_predict,ref,'GeoKeyDirectoryTag',info.GeoTIFFTags.GeoKeyDirectoryTag);


work_time=toc

