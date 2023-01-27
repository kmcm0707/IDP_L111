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
  line_follower();
}

void line_follower(){
  m1->run(FORWARD);
  m2->run(FORWARD);
  int error = 0; //error - turn left is +ve ,  turn right is -ve
  int last_error = 0;
  int I = 0;
  int basespeed = 200;
  const float k_i = 0.001;
  const float k_p = 30;
  const float k_d = 10;
  while(true){
    if(analogRead(left_line_follower) > linefollower_trigger &&  analogRead(right_line_follower) < linefollower_trigger) // Turn left - right wheel faster
    {
      error = 5;
    }
    if(analogRead(left_line_follower) < linefollower_trigger &&  analogRead(right_line_follower) > linefollower_trigger) // turn right - left wheel faster
    {
      error = -5;
    }
    if(analogRead(left_line_follower) < linefollower_trigger &&  analogRead(right_line_follower) < linefollower_trigger) // turn right - left wheel faster
    {
      error = last_error;
      if(error < 0){
        error = -10;
      }
      if (error > 0){
        error = 10;
      }
    }
    if(analogRead(left_line_follower) > linefollower_trigger &&  analogRead(right_line_follower) > linefollower_trigger) // turn right - left wheel faster
    {
      error = 0;
    }
    I = I + error;
    int D = error - last_error;
    last_error = error;

    int motorspeed = (int)(k_p*error + k_d * D + k_i * I);
    int leftspeed = basespeed - motorspeed;
    int rightspeed = basespeed + motorspeed;
    
    if(leftspeed < 0 || leftspeed > 255 || rightspeed < 0 || rightspeed > 255){
      if(leftspeed < 0){
        m1->run(BACKWARD);
        if(leftspeed > -255){
          m1->setSpeed(-leftspeed);
        } else {
          m1->setSpeed(255);
        }
      } else {
        m1->run(FORWARD);
        m1->setSpeed(255);
      }
      if(rightspeed < 0){
        m2->run(BACKWARD);
        if(rightspeed > -255){
          m2->setSpeed(-rightspeed);
        } else {
          m2->setSpeed(255);
        }
      } else {
        m2->run(FORWARD);
        m2->setSpeed(255);
      }
    }
    if(leftspeed > 0 && leftspeed < 255) {
      m1->run(FORWARD);
      m1->setSpeed(leftspeed);
    }
    if(rightspeed > 0 && rightspeed < 255){
      m2->run(FORWARD);
      m2->setSpeed(rightspeed);
    }
    
  }
  
}                    
