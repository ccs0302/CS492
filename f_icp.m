function [ER, Ticp, Ricp, t] = f_icp(data_name, S_pcd, T_pcd, opt_iter, matching_method)

    N = size(S_pcd,2);
    
    exp_name = data_name + "_" + matching_method + "_" + num2str(opt_iter);
    disp(exp_name);
    
    [Ricp, Ticp, ER, t] = icp(T_pcd, S_pcd, opt_iter, 'Matching', convertStringsToChars(matching_method), "ReturnAll", true);

    % Initialize and record video
    myVideo = VideoWriter(convertStringsToChars(exp_name)); %open video file
    myVideo.FrameRate = 10;  %can adjust this, 5 - 10 works well for me
    open(myVideo)

    show_fg = figure;
    for k = 1:opt_iter+1
        figure(show_fg);
        Sinc = Ricp(:,:,k) * S_pcd + repmat(Ticp(:,:,k), 1, N);
        plot3(T_pcd(1,:),T_pcd(2,:),T_pcd(3,:),'bo',Sinc(1,:),Sinc(2,:),Sinc(3,:),'r.');
        set(gca,'Color','k')
        xlabel('x'); ylabel('y'); zlabel('z');
        title("ICP result (iter: "+int2str(k)+")");
        drawnow;
        pause(0.02);
        
        frame = getframe(gcf);
        writeVideo(myVideo, frame);
    end
    close(myVideo)

    % Plot Target_pcd points (blue) and Source_pcd points (red)
    qualitative_result_fg = figure;
    figure(qualitative_result_fg);
    subplot(1,2,1);
    plot3(T_pcd(1,:),T_pcd(2,:),T_pcd(3,:),'bo',S_pcd(1,:),S_pcd(2,:),S_pcd(3,:),'r.');
    set(gca,'Color','k')
    axis equal;
    xlabel('x'); ylabel('y'); zlabel('z');
    title('Original 2 point cloud sets');

    subplot(1,2,2);
    plot3(T_pcd(1,:),T_pcd(2,:),T_pcd(3,:),'bo',Sinc(1,:),Sinc(2,:),Sinc(3,:),'r.');
    set(gca,'Color','k')
    axis equal;
    xlabel('x'); ylabel('y'); zlabel('z');
    title('ICP result');

end