clear all
% Загружаем анализируемую аудиозапись голоса летучих мышей
%bat1.wav
%bat2.wav
%bat_h_f.flac
%bat_h_f_2.wav
%bat_h_f_3.wav
[y,Fs_audio] = audioread('bat_h_f_3.wav');
y = y(:,1); %Если стерео-аудиозапись, то 
%sound(y,8192); %На сниженной частоте писк слышен
%Чем ниже частота, тем медленнее воспроизводится аудио, тем ниже и
%басовитее становится писк

Ny = length(y); %Длина последовательности
nsc = 1024;%floor(Ny/12); %Длина окна
nov = 512;%floor(nsc/8); %Значение перекрытия
nff = max(256,2^nextpow2(nsc)); %Длина ДПФ
%{
figure(4);

subplot(2,1,1);
spectrogram(y,blackman(nsc),nov,nff,'yaxis',Fs_audio);
%xlabel('Время (в миллисекундах)');
title("Спектрограмма");

subplot(2,1,2);
plot(y);
xlabel('Время (в отсчётах)');
title("Осциллограмма");
%}
%% Нормальное распределение

%n = normrnd(mean(y),var(sqrt(y)),size(y,1),1); %Генерируем числа для нормального распределения
%figure(3);
%hold on;
%hy = histogram(y);
%hn = histogram(n);

%legend('Гистограмма писка','Гистограмма нормального распределения');
histfit(y,ceil(abs(sqrt(size(y,1)))),'norm');

%% Спектрограммы
%{
figure(2);

subplot(2,2,1);
spectrogram(y,rectwin(nsc),nov,nff,'yaxis',Fs_audio);
title("Прямоугольное окно");

subplot(2,2,2);
spectrogram(y,triang(nsc),nov,nff,'yaxis',Fs_audio);
title("Треугольное окно");

subplot(2,2,3);
spectrogram(y,hann(nsc),nov,nff,'yaxis',Fs_audio);
title("Окно Ханна");

subplot(2,2,4);
spectrogram(y,blackman(nsc),nov,nff,'yaxis',Fs_audio);
title("Окно Блэкмана");
%}
%%
%Сделаем водопад (для этого разобьём аудиофайл на равные отрезки и для
%каждого найдём преобразование Фурье, после чего запишем всё в двумерный
%массив)
%wf_buf = zeros(fix(length(y)/N),N+1); %Инициализируем массивы
%fft_wf_buf = zeros(fix(length(y)/N),N+1);
%for i = 1:fix(length(y)/N)
  %wf_buf(i,:) = y(N*i-N+1:N*i+1); %Нарезаем последовательность в двумерный массив (временная область)
  %fft_wf_buf(i,:) = fft(wf_buf(i,:)); %Двумерный массив в частотной области
%end
%%
%nu = ((1:N)/N)*250e3; %Нормированная частота
%nu = ((1:N)/N-0.5)*250e3;
%w = hann(N); %Создаём отсчёты окна (Ханна)
%N = 4096*2; %Длина преобразовнаия Фурье буфера аудиозаписи
%buf1 = y(1:N,1); %Выбираем отрезок из аудиозаписи
%buf1_w = buf1.*w; %Умножаем отрезок аудиозаписи на окно
%fft_buf1 = fft(buf1); %Спектр отрезка без окна
%fft_buf1_w = fft(buf1_w); %Спектр отрезка с окном

%buf2 = y(N:2*N-1,1); %Выбираем другой отрезок из аудиозаписи
%buf2_w = buf2.*w; %Умножаем другой отрезок аудиозаписи на окно
%fft_buf2 = fft(buf2); %Спектр другого отрезка без окна
%fft_buf2_w = fft(buf2_w); %Спектр другого отрезка с окном
%% График сигнала во временной области
%figure(1);
%plot(y);
%title("Весь сигнал во временной области");
%xlabel('Время (в отсчётах)');
%ylabel('Амплитуда');
%grid;
%%
%figure(2);
%subplot(2,2,1)
%semilogy(nu,abs(fftshift(fft_buf1)));
%title("Спектр первого отрезка аудиозаписи без окна");
%xlabel('Нормированная частота');
%ylabel('Амплитуда');
%grid;

%subplot(2,2,2)
%semilogy(nu,abs(fftshift(fft_buf1_w)));
%title("Спектр первого отрезка аудиозаписи с окном");
%xlabel('Нормированная частота');
%ylabel('Амплитуда');
%grid;

%subplot(2,2,3)
%semilogy(nu,abs(fftshift(fft_buf2)));
%title("Спектр второго отрезка аудиозаписи без окна");
%xlabel('Нормированная частота');
%ylabel('Амплитуда');
%grid;

%subplot(2,2,4)
%semilogy(nu,abs(fftshift(fft_buf2_w)));
%title("Спектр второго отрезка аудиозаписи с окном");
%xlabel('Нормированная частота');
%ylabel('Амплитуда');
%grid;
