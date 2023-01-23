#define left_line_follower A4 
#define right_line_follower A5

#include <Adafruit_MotorShield.h>
int linefollower_trigger = 200;
Adafruit_MotorShield AFMS = Adafruit_MotorShield();

Adafruit_DCMotor m1 = AFMS.getMotor(1);
Adafruit_DCMotor m2 = AFMS.getMotor(2);
Adafruit_DCMotor m3 = AFMS.getMotor(3);
Adafruit_DCMotor m3 = AFMS.getMotor(4);

void setup() {
  Serial.begin(9600);

  //line folllower
  // put your setup code here, to run once:
  pinMode(left_line_follower,INPUT);
  pinMode(right_line_follower,INPUT);

  //motor setup
  
  if (!AFMS.begin()) {         // create with the default frequency 1.6KHz
  // if (!AFMS.begin(1000)) {  // OR with a different frequency, say 1KHz
    Serial.println("Could not find Motor Shield. Check wiring.");
    while (1);
  }
  Serial.println("Motor Shield found.");

  // Set the speed to start, from 0 (off) to 255 (max speed)
  myMotor->setSpeed(150);
  myMotor->run(FORWARD);
  // turn on motor
  myMotor->run(RELEASE);
}

void loop() {
  uint8_t i;

  Serial.print("tick");

  myMotor->run(FORWARD);
  for (i=0; i<255; i++) {
    myMotor->setSpeed(i);
    delay(10);
  }
  for (i=255; i!=0; i--) {
    myMotor->setSpeed(i);
    delay(10);
  }

  Serial.print("tock");

  myMotor->run(BACKWARD);
  for (i=0; i<255; i++) {
    myMotor->setSpeed(i);
    delay(10);
  }
  for (i=255; i!=0; i--) {
    myMotor->setSpeed(i);
    delay(10);
  }

  Serial.print("tech");
  myMotor->run(RELEASE);
  delay(1000);
}
