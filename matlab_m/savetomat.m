function savetomat(myfilefolder,filename,n,savepath)
   
body1 = read_skeleton_file([myfilefolder,filename])
num=length(body1)
%%fprintf('x: %f \n',body1(1).bodies.joints(1).x)
% global x
% global y
% % global z
% body1.bodies(1).joints.x==(num,25)
x=zeros(num,25);
for i=1:num
    for j=1:25
        x(i,j)=body1(i).bodies(1).joints(j).x;
    end
end

y=zeros(num,25);

for i=1:num
    for j=1:25
        y(i,j)=body1(i).bodies(1).joints(j).y;
    end
end
z=zeros(num,25);
for i=1:num
    for j=1:25
        z(i,j)=body1(i).bodies(1).joints(j).z;
    end
end
filename
myend=findstr(filename,'.')
myend=myend(1)
chr=[savepath,filename(1:myend-1),'.mat']
chr
save ( chr,'x','y','z');
end