#include <Wire.h>
#include <Adafruit_MotorShield.h>

Adafruit_MotorShield AFMS = Adafruit_MotorShield();

Adafruit_DCMotor *m1 = AFMS.getMotor(1); //left
Adafruit_DCMotor *m2 = AFMS.getMotor(2); //right    

// pins
int left_line_follower = 0;
int right_line_follower = 1;

int status_check = 0; //set to 0 at start set to 1 during line following


int speed;
String speed_str;

// defines pins numbers
const int trigPin = 9;
const int echoPin = 10;
long duration;
int distance;

void setup() {
  // put your setup code here, to run once:

   if (!AFMS.begin()) {
    while (1);
  }

  Serial.begin(9600);

}

void loop() {
  delay(1000);
  line_follower();
}


void line_follower(){
  m1->run(FORWARD);
  m2->run(FORWARD);
  int error = 0; //error - turn left is +ve ,  turn right is -ve
  int last_error = 0;
  int I = 0;
  int basespeed = 200;
  const float k_i = 0.009; // 0.001
  const float k_p = 20;  // 30
  const float k_d = 15;  // 10
  while(true){
    Serial.println(error);
    
    if(digitalRead(left_line_follower) &&  ! digitalRead(right_line_follower)) // Turn left - right wheel faster
    {
      error = 5; // 5
    }
    if(!digitalRead(left_line_follower) &&  digitalRead(right_line_follower)) // turn right - left wheel faster
    {
      error = -5; // -5
    }
    if(!digitalRead(left_line_follower) &&  !digitalRead(right_line_follower)) // nither on line
    {
        error = last_error;
        if(error < 0){
          error = 10;
        }
        if (error > 0){
          error = -10;
        }
    }
    if(digitalRead(left_line_follower)  &&  digitalRead(right_line_follower)) // both on line no error
    {
      error = 0;
    }
    I = I + error;
    int D = error - last_error;
    last_error = error;

    int motorspeed = (int)(k_p*error + k_d * D + k_i * I);
    int leftspeed = basespeed - motorspeed;
    int rightspeed = basespeed + motorspeed;

    if(leftspeed > 0){
      m1->run(FORWARD);
    } else {
      m1->run(BACKWARD);
    }

    if(rightspeed > 0){
      m2->run(FORWARD);
    } else {
      m2->run(BACKWARD);
    }

    if (leftspeed > 255){
      m1->setSpeed(255);
    }
    if (leftspeed < 0){
      m1->setSpeed(0);
    }

    if (rightspeed > 255){
      m2->setSpeed(255);
    }
    if (rightspeed < 0){
      m2->setSpeed(0);
    }
  }


}
