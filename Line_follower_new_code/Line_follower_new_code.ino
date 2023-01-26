#include <Wire.h>
#include <Adafruit_MotorShield.h>
#include "utility/Adafruit_MS_PWMServoDriver.h"

Adafruit_MotorShield AFMS = Adafruit_MotorShield();

Adafruit_DCMotor *m1 = AFMS.getMotor(1); //left
Adafruit_DCMotor *m2 = AFMS.getMotor(2); //right    

int left_line_follower = A0;
int right_line_follower = A1;

const int min_speed = 0;
const int max_speed = 255;

int linefollower_trigger = 500;
void setup() {
  // put your setup code here, to run once:

   if (!AFMS.begin()) {
    Serial.println("Could not find Motor Shield. Check wiring.");
    while (1);
  }
  
}

void loop() {
  // put your main code here, to run repeatedly:

}

void line_follower(){
  int error = 0; //error - turn left is +ve ,  turn right is -ve
  int last_error = 0;
  int integral = 0;
  int basespeed = 170;
  const int k_i = 0.0008;
  const int k_p = 0.07;
  const int k_d = 0.6;
  while(true){
    if(analogRead(left_line_follower) > linefollower_trigger &&  analogRead(right_line_follower) < linefollower_trigger) // Turn left - right wheel faster
    {
      error = 5;
    }
    if(analogRead(left_line_follower) < linefollower_trigger &&  analogRead(right_line_follower) > linefollower_trigger) // turn right - left wheel faster
    {
      error = -5;
    }
    if(analogRead(left_line_follower) > linefollower_trigger &&  analogRead(right_line_follower) > linefollower_trigger) // turn right - left wheel faster
    {
      error = 0;
    }
    I = I + error;
    int D = error - last_error;
    last_error = error;

    int motorspeed = (int)(k_p*error + k_d * D + k_i * I);
    int leftspeed = basespeed - motorspeed
    int rightspeed = basespeed + motorspeed
    
  }
  
}

`                       
