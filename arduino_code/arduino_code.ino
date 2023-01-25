#include <Wire.h>
#include <Adafruit_MotorShield.h>
#include "utility/Adafruit_MS_PWMServoDriver.h"

Adafruit_MotorShield AFMS = Adafruit_MotorShield();

Adafruit_DCMotor *m1 = AFMS.getMotor(1); //left
Adafruit_DCMotor *m2 = AFMS.getMotor(2); //right

int left_line_follower = A0;
int right_line_follower = A1;
int outer_left_follower = A2;
int outer_right_follower = A3;
//int left_sensorValue = 0;
//int right_sensorValue = 0;
int linefollower_trigger = 600;

int detect_blue = 1;
int detect_red = 2;
int read_blue = 3;
int read_red = 4;

void setup() {
  Serial.begin(9600);

  //line folllower
  // put your setup code here, to run once:
  
  //motor setup
  
  if (!AFMS.begin()) {         // create with the default frequency 1.6KHz
  // if (!AFMS.begin(1000)) {  // OR with a different frequency, say 1KHz
    Serial.println("Could not find Motor Shield. Check wiring.");
    while (1);
  }
  Serial.println("Motor Shield found.");

  // Set the speed to start, from 0 (off) to 255 (max speed)
  m1->setSpeed(250);
  m2->setSpeed(250);
  m1->run(FORWARD);
  m2->run(FORWARD);


  //colour detection
  int detect_blue = 1;
  int detect_red = 2;
  int output_blue = 3;
  int output_red = 4;
  pinMode(detect_blue, INPUT);
  pinMode(detect_red, INPUT);
  pinMode(output_blue, OUTPUT);
  pinMode(output_red, OUTPUT);
}

void loop() {
  uint8_t i;
  // left_sensorValue = analogRead(left_line_follower);
  // Serial.println(left_sensorValue);
  line_follower_feedback();
}


/*
 * old code
 * void line_following(){
  m1->setSpeed(150);
  m2->setSpeed(150);
  if(analogRead(left_line_follower) > linefollower_trigger &&  analogRead(right_line_follower) > linefollower_trigger) // Move Forward
  {
  m1->run(FORWARD);
  m2->run(FORWARD);
  }
  
  if(analogRead(left_line_follower) > linefollower_trigger &&  analogRead(right_line_follower) < linefollower_trigger) // Turn right
  {
  m1->run(BACKWARD);
  m2->run(FORWARD);
  }
  
  if(analogRead(left_line_follower) < linefollower_trigger &&  analogRead(right_line_follower) > linefollower_trigger) // turn left
  {
  m1->run(FORWARD);
  m2->run(BACKWARD);
  }
  
  if(analogRead(left_line_follower) < linefollower_trigger &&  analogRead(right_line_follower) < linefollower_trigger) // stop !(digitalRead(LS)) && !(digitalRead(RS))
  {
  m1->run(RELEASE);
  m2->run(RELEASE);
  }
}*/

void line_follower_feedback() {
  int m1_currentspeed = 180;
  int m2_currentspeed = 180;
  int k_pos = 1;
  int k_negetive = -2;
  m1->setSpeed(m1_currentspeed);
  m2->setSpeed(m2_currentspeed);
  m1->run(FORWARD);
  m2->run(FORWARD);
  while(true){
    if(analogRead(left_line_follower) > linefollower_trigger &&  analogRead(right_line_follower) > linefollower_trigger) // Move Forward and do nothing
    {

    }
    if(analogRead(left_line_follower) > linefollower_trigger &&  analogRead(right_line_follower) < linefollower_trigger) // Turn left
    {
      m1_currentspeed = m1_currentspeed + k_negetive;
      m2_currentspeed = m2_currentspeed + k_pos;
    }
    
    if(analogRead(left_line_follower) < linefollower_trigger &&  analogRead(right_line_follower) > linefollower_trigger) // turn right
    {
      m1_currentspeed = m1_currentspeed + k_pos;
      m2_currentspeed = m2_currentspeed + k_negetive;
    }
    
    if(analogRead(left_line_follower) < linefollower_trigger &&  analogRead(right_line_follower) < linefollower_trigger) // stop !(digitalRead(LS)) && !(digitalRead(RS))
    {
      break;
    }
    if(analogRead(outer_left_follower) > linefollower_trigger ||  analogRead(outer_right_follower) > linefollower_trigger) // stop !(digitalRead(LS)) && !(digitalRead(RS))
    {
      break;
    }
    m1->setSpeed(m1_currentspeed);
    m2->setSpeed(m2_currentspeed);
  }
}

void colour_detection(){
  int detect_blue = 1;
  int detect_red = 2;
  int output_blue = 3;
  int output_red = 4;
  if(digitalRead(detect_blue) == 1){
    digitalWrite(output_blue, HIGH);
    delay(2000);
    digitalWrite(output_blue, LOW);
  }
  if(digitalRead(detect_red) == 1){
    digitalWrite(output_red, HIGH);
    delay(2000);
    digitalWrite(output_red, LOW);
  }
}
