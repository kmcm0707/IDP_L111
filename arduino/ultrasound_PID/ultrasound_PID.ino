// Code that was used to test the Ultrasound PID controller
#include <HCSR04.h>
#include <Wire.h>
#include <Adafruit_MotorShield.h>
#include "utility/Adafruit_MS_PWMServoDriver.h"

Adafruit_MotorShield AFMS = Adafruit_MotorShield();

Adafruit_DCMotor *m1 = AFMS.getMotor(1); //left
Adafruit_DCMotor *m2 = AFMS.getMotor(2); //right 
const int trigPin = 6;
const int echoPin = 7;
// defines variables
long duration;
int distance;
UltraSonicDistanceSensor distancesensor(trigPin, echoPin);

void setup() {
  Serial.begin(9600);
   if (!AFMS.begin()) {
    while (1);
  }// Starts the serial communication
}
void loop() {
  ultrasound_follower();
}
void ultrasound_follower(){
  m1->run(FORWARD);
  m2->run(FORWARD);
  int error = 0; //error - turn left is +ve ,  turn right is -ve
  int last_error = 0;
  int I = 0;
  int basespeed = 180;
  const float k_i = 0.0001;
  const float k_p = 30;
  const float k_d = 10;
  while(distancesensor.measureDistanceCm() < 60){
    error = -(7.53 - distancesensor.measureDistanceCm()); 
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
      } else{
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
