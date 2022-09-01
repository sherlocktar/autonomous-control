#include <Servo.h>

//int error;
int sudut;
Servo servoku;
char data;
void setup() {
  servoku.attach(10);
  Serial.begin(9600);
}

void loop() {
  if (Serial.available() > 0) {
    data = Serial.read();
    Serial.println(data);
    /////tengah = 90
    while (data == 'a') {
      servoku.write(90);
      data = Serial.read();
      //Serial.println(data);
    }
    //////kanan 90 ->
    while (data == 'b') {
      servoku.write(102);
      data = Serial.read();
      //Serial.println(data);
    }
    while (data == 'c') {
      servoku.write(108);
      data = Serial.read();
      //Serial.println(data);
    }
    while (data == 'd') {
      servoku.write(114);
      data = Serial.read();
      //Serial.println(data);
    }
    while (data == 'e') {
      servoku.write(120);
      data = Serial.read();
      //Serial.println(data);
    }
    while (data == 'f') {
      servoku.write(126);
      data = Serial.read();
      //Serial.println(data);
    }
    while (data == 'g') {
      servoku.write(132);
      data = Serial.read();
      //Serial.println(data);
    }
    while (data == 'h') {
      servoku.write(138);
      data = Serial.read();
      //Serial.println(data);
    }
    while (data == 'i') {
      servoku.write(144);
      data = Serial.read();
      //Serial.println(data);
    }
    while (data == 'j') {
      servoku.write(150);
      data = Serial.read();
      //Serial.println(data);
    }
    while (data == 'k') {
      servoku.write(156);
      data = Serial.read();
      //Serial.println(data);
    }
    while (data == 'l') {
      servoku.write(162);
      data = Serial.read();
      //Serial.println(data);
    }
    while (data == 'm') {
      servoku.write(168);
      data = Serial.read();
      //Serial.println(data);
    }
    while (data == 'n') {
      servoku.write(174);
      data = Serial.read();
      //Serial.println(data);
    }
    while (data == 'o') {
      servoku.write(180);
      data = Serial.read();
      //Serial.println(data);
    }
    //////////kiri
    while (data == 'B') {
      servoku.write(78);
      data = Serial.read();
      //Serial.println(data);
    }
    while (data == 'C') {
      servoku.write(72);
      data = Serial.read();
      //Serial.println(data);
    }
    while (data == 'D') {
      servoku.write(66);
      data = Serial.read();
      //Serial.println(data);
    }
    while (data == 'E') {
      servoku.write(60);
      data = Serial.read();
      //Serial.println(data);
    }
    while (data == 'F') {
      servoku.write(54);
      data = Serial.read();
      //Serial.println(data);
    }
    while (data == 'G') {
      servoku.write(48);
      data = Serial.read();
      //Serial.println(data);
    }
    while (data == 'H') {
      servoku.write(42);
      data = Serial.read();
      //Serial.println(data);
    }
    while (data == 'I') {
      servoku.write(36);
      data = Serial.read();
      //Serial.println(data);
    }
    while (data == 'J') {
      servoku.write(30);
      data = Serial.read();
      //Serial.println(data);
    }
    while (data == 'K') {
      servoku.write(24);
      data = Serial.read();
      //Serial.println(data);
    }
    while (data == 'L') {
      servoku.write(18);
      data = Serial.read();
      //Serial.println(data);
    }
    while (data == 'M') {
      servoku.write(12);
      data = Serial.read();
      //Serial.println(data);
    }
    while (data == 'N') {
      servoku.write(6);
      data = Serial.read();
      //Serial.println(data);
    }
    while (data == 'O') {
      servoku.write(0);
      data = Serial.read();
      //Serial.println(data);
    }
  }

}
