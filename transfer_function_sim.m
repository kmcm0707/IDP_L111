ki = 0.001;
kd = 30;
kp = 50;
sys = tf([5*kd 5*kp 5*ki],[1 (5*kd) (5*kp) 5*ki])
subplot(2,1,1)
step(sys)
xlim([0 1.5])