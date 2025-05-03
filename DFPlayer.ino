#include <GD3300.h>

GD3300 speakerA;  // Player on Serial1
GD3300 speakerB;  // Player on Serial2

const int piezoPins[6] = {A1, A2, A3, A4, A5};


void setup() {
  Serial.begin(9600);

  Serial1.begin(9600);  // TX1=18, RX1=19
  Serial2.begin(9600);  // TX2=16, RX2=17
  delay(2000);          // Let modules boot

  speakerA.begin(Serial1);
  speakerB.begin(Serial2);

  speakerA.setVol(30);
  speakerB.setVol(30);

  
  speakerA.playSL(1);  // Play 0001.mp3 on Speaker A
  delay (500);
  speakerB.playSL(1);  // Play 0001.mp3 on Speaker B
}

void loop() {
  unsigned long t = micros();
  for (int i = 0; i < 5; i++) {
    Serial.print(analogRead(piezoPins[i]));
    Serial.print(",");
  }
  Serial.println(t);
  
}
