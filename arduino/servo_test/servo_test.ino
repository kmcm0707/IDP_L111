#include <Servo.h>

Servo servo1;

int target = 120;


void setup() {
  // put your setup code here, to run once:
  servo1.attach(9);

}

void loop() {
  // put your main code here, to run repeatedly:
  for (int i = 0; i < target; i++) {
    servo1.write(i);
    delay(15);
  }
  delay(100);
  for (int i = target; i > 0; i--) {
    servo1.write(i);
    delay(15);
  }
  delay(100);

}
