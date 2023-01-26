#include <Wire.h>
#include <Adafruit_MotorShield.h>
#include "utility/Adafruit_MS_PWMServoDriver.h"

Adafruit_MotorShield AFMS = Adafruit_MotorShield();

Adafruit_DCMotor *m1 = AFMS.getMotor(1); //left
Adafruit_DCMotor *m2 = AFMS.getMotor(2); //right    

int left_line_follower = A0;
int right_line_follower = A1;

int linefollower_trigger = 500;
void setup() {
  // put your setup code here, to run once:

   if (!AFMS.begin()) {
    Serial.println("Could not find Motor Shield. Check wiring.");
    while (1);
  }
  
}

void loop() {
  delay(1000);
  // put your main code here, to run repeatedly:
  line_follower()
}

void line_follower(){
  m1->run(FORWARD);
  m2->run(FORWARD);
  int error = 0; //error - turn left is +ve ,  turn right is -ve
  int last_error = 0;
  int integral = 0;
  int basespeed = 170;
  const float k_i = 0.0008;
  const float k_p = 0.07;
  const float k_d = 0.6;
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
    
    if(leftspeed < min_speed || leftspeed > max_speed){
      if(leftspeed < min_speed){
        m1->setSpeed(0);
      } else {
        m1->setSpeed(255);
      }
    } else if (rightspeed < min_speed || rightspeed > max_speed) {
      if(rightspeed < min_speed){
        m2->setSpeed(0);
      } else {
        m2->setSpeed(255);
      }
    } else {
      m1->setSpeed(leftspeed);
      m2->setSpeed(rightspeed);
    }
    
  }
  
}                    
