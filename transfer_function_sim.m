ki = 0.0001;
kd = 10;
kp = 30;
sys = tf([5*kd 5*kp 5*ki],[1 (5*kd) (5*kp) 5*ki])
subplot(2,1,1)
step(sys)
xlim([0 1.5])