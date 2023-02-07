int left_line_follower = 0;
int right_line_follower = 1;

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
}

void loop() {
  Serial.println(digitalRead(left_line_follower));
  Serial.println(digitalRead(right_line_follower));
  delay(100); 
}
