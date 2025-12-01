#include <M5Unified.h>
#include <math.h>

// ---------- CONFIG ANN (4 -> 8 -> 1) ----------
const int INPUT_DIM  = 4;
const int HIDDEN_DIM = 8;

// StandardScaler (valeurs venant de ton entraînement)
float feature_mean[INPUT_DIM] = {
  2.382548f, 4.647696f, 2.584395f, 9.614640f
};

float feature_std[INPUT_DIM] = {
  1.640936f, 5.726933f, 1.199374f, 7.415529f
};

const float LEARNING_RATE      = 0.05f;
const float DECISION_THRESHOLD = 0.60f;

// Poids initiaux du réseau
float W1[HIDDEN_DIM][INPUT_DIM] = {
  {0.5,-0.3,0.2,0.1},
  {-0.4,0.6,0.1,-0.2},
  {0.3,0.3,-0.5,0.2},
  {0.1,0.2,0.4,-0.3},
  {-0.2,-0.1,0.5,0.4},
  {0.6,-0.2,0.3,0.1},
  {-0.3,0.4,-0.1,0.2},
  {0.2,0.1,0.2,-0.4}
};

float b1[HIDDEN_DIM] = {0,0.1,-0.1,0.05,0,-0.05,0.1,0};
float W2[HIDDEN_DIM] = {0.4,-0.3,0.5,-0.2,0.3,-0.4,0.2,0.1};
float b2             = 0.0;

// ---------- STRUCTURES UI ----------
struct Key { int x,y,w,h; char label; };
Key keys[12];

String   currentCode = "";
uint32_t timeStamps[4];
int      digitIndex  = 0;

int currentUser = 0;   // 0 ou 1

enum Mode { MODE_TRAIN=0, MODE_EVAL=1, MODE_TEST=2 };
Mode currentMode = MODE_TRAIN;

uint32_t evalTotal   = 0;
uint32_t evalCorrect = 0;

// Layout écran
const int HEADER_H  = 40;   // bandeau User/Mode
const int CODE_H    = 40;   // bandeau Code
const int INFO_H    = 150;  // zone du bas (textes + graph)
const int CONTROL_H = 40;   // barre du bas
const int KEYAREA_MAX_H = 220; // hauteur MAX du clavier (plus grand que avant)

// ---------- ANN ----------
float relu(float x){ return (x>0)?x:0; }
float sigmoid(float x){ return 1.0f/(1.0f+expf(-x)); }

float ann_predict(float t1,float t2,float t3,float T){
  float x[4]={t1,t2,t3,T};
  for(int i=0;i<4;i++)
    x[i]=(x[i]-feature_mean[i])/feature_std[i];

  float h[8];
  for(int j=0;j<8;j++){
    float s=b1[j];
    for(int i=0;i<4;i++) s+=W1[j][i]*x[i];
    h[j]=relu(s);
  }

  float s=b2;
  for(int j=0;j<8;j++) s+=W2[j]*h[j];
  return sigmoid(s);  // proba d'être user1
}

float ann_train_sample(float t1,float t2,float t3,float T,int target){
  float x[4]={t1,t2,t3,T};
  for(int i=0;i<4;i++)
    x[i]=(x[i]-feature_mean[i])/feature_std[i];

  float z1[8], h[8];
  for(int j=0;j<8;j++){
    float s=b1[j];
    for(int i=0;i<4;i++) s+=W1[j][i]*x[i];
    z1[j]=s;
    h[j]=relu(s);
  }

  float z2=b2;
  for(int j=0;j<8;j++) z2+=W2[j]*h[j];
  float y=sigmoid(z2);

  float t=(float)target;
  float dL_dy  = y - t;
  float dY_dZ2 = y*(1-y);
  float dL_dZ2 = dL_dy*dY_dZ2;

  float dL_dW2[8], dL_dZ1[8];
  for(int j=0;j<8;j++){
    dL_dW2[j] = dL_dZ2*h[j];
    float dHdZ = (z1[j]>0)?1:0;
    dL_dZ1[j] = dL_dZ2*W2[j]*dHdZ;
  }

  // update sortie
  b2 -= LEARNING_RATE*dL_dZ2;
  for(int j=0;j<8;j++)
    W2[j] -= LEARNING_RATE*dL_dW2[j];

  // update couche cachée
  for(int j=0;j<8;j++){
    for(int i=0;i<4;i++)
      W1[j][i] -= LEARNING_RATE*dL_dZ1[j]*x[i];
    b1[j] -= LEARNING_RATE*dL_dZ1[j];
  }

  return y; // proba user1 avant update
}

// ---------- UI HELPERS ----------
const char* modeName(Mode m){
  if(m==MODE_TRAIN) return "TRAIN";
  if(m==MODE_EVAL)  return "EVAL";
  return "TEST";
}

void drawControlBar(){
  auto &d=M5.Display;
  int sw=d.width(), sh=d.height();

  d.fillRect(0,sh-CONTROL_H,sw,CONTROL_H,TFT_DARKGREY);
  d.setTextColor(TFT_WHITE,TFT_DARKGREY);
  d.setTextSize(2);
  d.setCursor(5,sh-CONTROL_H+10);
  d.print("G:Effacer  D:");
  d.print(modeName(currentMode));
}

void drawKeyboard(){
  auto &d=M5.Display;
  int sw=d.width();
  int sh=d.height();

  d.fillScreen(TFT_BLACK);

  // HEADER
  d.fillRect(0,0,sw,HEADER_H,TFT_DARKGREY);
  d.setTextColor(TFT_WHITE,TFT_DARKGREY);
  d.setTextSize(2);
  d.setCursor(5,HEADER_H/2-7);
  d.printf("User:%d  %s",currentUser,modeName(currentMode));

  // CODE
  d.fillRect(0,HEADER_H,sw,CODE_H,TFT_NAVY);
  d.setTextColor(TFT_WHITE,TFT_NAVY);
  d.setTextSize(3);
  d.setCursor(5,HEADER_H + CODE_H/2 - 10);
  d.print("Code:");
  d.print(currentCode);

  // CALCUL ZONES
  int infoY = sh - CONTROL_H - INFO_H;   // top zone info
  int keyTop = HEADER_H + CODE_H;        // start clavier
  int maxPossibleKeyH = infoY - keyTop;  // ce qui reste

  if (maxPossibleKeyH < 40) maxPossibleKeyH = 40;
  int keyAreaH = maxPossibleKeyH;
  if (keyAreaH > KEYAREA_MAX_H) keyAreaH = KEYAREA_MAX_H;  // LIMITATION

  // CLAVIER (4x3) — cases + chiffres plus gros
  int rows=4, cols=3;
  int marginX=10, marginY=6;
  int keyW=(sw - 2*marginX - (cols-1)*marginX)/cols;
  int keyH=(keyAreaH - (rows-1)*marginY)/rows;

  char labels[12] = {
    '1','2','3',
    '4','5','6',
    '7','8','9',
    ' ','0',' '
  };

  int idx=0;
  for(int r=0;r<rows;r++){
    for(int c=0;c<cols;c++){
      int x=marginX + c*(keyW+marginX);
      int y=keyTop + r*(keyH+marginY);
      char lab = labels[idx];

      keys[idx] = {x,y,keyW,keyH,lab};

      if(lab!=' '){
        d.fillRect(x,y,keyW,keyH,TFT_DARKGREY);
        d.drawRect(x,y,keyW,keyH,TFT_WHITE);
        d.setTextColor(TFT_WHITE,TFT_DARKGREY);
        d.setTextSize(3);  // CHIFFRES PLUS GRANDS
        d.setCursor(x + keyW/2 - 8, y + keyH/2 - 12);
        d.print(lab);
      }
      idx++;
    }
  }

  // ZONE INFO
  d.fillRect(0,infoY,sw,INFO_H,TFT_BLACK);

  drawControlBar();
}

void updateHeader(){
  auto &d=M5.Display;
  int sw=d.width();

  // HEADER
  d.fillRect(0,0,sw,HEADER_H,TFT_DARKGREY);
  d.setTextColor(TFT_WHITE,TFT_DARKGREY);
  d.setTextSize(2);
  d.setCursor(5,HEADER_H/2-7);
  d.printf("User:%d  %s",currentUser,modeName(currentMode));

  // CODE
  d.fillRect(0,HEADER_H,sw,CODE_H,TFT_NAVY);
  d.setTextColor(TFT_WHITE,TFT_NAVY);
  d.setTextSize(3);
  d.setCursor(5,HEADER_H + CODE_H/2 - 10);
  d.print("Code:");
  d.print(currentCode);

  drawControlBar();
}

// ---------- TRAITEMENT D'UN CODE ----------
void processCode(){
  if(digitIndex!=4) return;

  float t1=(timeStamps[1]-timeStamps[0])/1000.0f;
  float t2=(timeStamps[2]-timeStamps[1])/1000.0f;
  float t3=(timeStamps[3]-timeStamps[2])/1000.0f;
  float T =(timeStamps[3]-timeStamps[0])/1000.0f;

  float p;
  if(currentMode==MODE_TRAIN)
       p = ann_train_sample(t1,t2,t3,T,currentUser);
  else p = ann_predict(t1,t2,t3,T);

  int pred = (p>=0.5f)?1:0;

  if(currentMode==MODE_EVAL){
    evalTotal++;
    if(pred==currentUser) evalCorrect++;
  }

  auto &d=M5.Display;
  int sw=d.width(), sh=d.height();

  // -------- ZONE INFO REMONTÉE --------
  int infoY = sh - CONTROL_H - INFO_H - 20;   // -20 pour monter la ligne jaune

  // LIGNE TEMPS (JAUNE) — plus haut
  d.fillRect(0,infoY,sw,24,TFT_BLACK);
  d.setTextColor(TFT_YELLOW,TFT_BLACK);
  d.setTextSize(2);
  d.setCursor(0,infoY+4);
  d.printf("t1=%.2f  t2=%.2f  t3=%.2f  T=%.2f",t1,t2,t3,T);

  // Texte + graph
  d.fillRect(0,infoY+24,sw,INFO_H-24,TFT_BLACK);
  d.setTextSize(2);
  d.setCursor(0,infoY+30);

  if(currentMode==MODE_EVAL){
    float probU1 = p * 100.0f;
    float probU0 = (1.0f - p) * 100.0f;
    float acc    = (evalTotal>0) ? (100.0f * (float)evalCorrect / (float)evalTotal) : 0.0f;
    const char* choisi = (pred==0) ? "User0" : "User1";

    d.setTextColor(TFT_WHITE,TFT_BLACK);
    d.printf("U0:%3.0f%%   U1:%3.0f%%", probU0, probU1);

    d.setCursor(0,infoY+52);
    d.printf("Je choisis %s | Acc:%3.0f%%", choisi, acc);

    // ---------- GRAPHIQUE BARRES ----------
    int barBaseY = infoY + INFO_H - 40;  // bas des barres (remontées)
    int barMaxH  = 40;
    int barWidth = 30;
    int barGap   = 25;

    int h0 = (int)(barMaxH * (probU0 / 100.0f));
    int h1 = (int)(barMaxH * (probU1 / 100.0f));

    int x0 = 20;
    int x1 = x0 + barWidth + barGap;

    // Efface zone graph
    d.fillRect(0, barBaseY - barMaxH - 20, sw, barMaxH + 30, TFT_BLACK);

    // Barre U0
    d.fillRect(x0, barBaseY - h0, barWidth, h0, TFT_BLUE);
    // Barre U1
    d.fillRect(x1, barBaseY - h1, barWidth, h1, TFT_RED);

    // Légendes
    d.setTextSize(2);
    d.setTextColor(TFT_WHITE,TFT_BLACK);
    d.setCursor(x0, barBaseY - barMaxH - 18);
    d.print("U0");
    d.setCursor(x1, barBaseY - barMaxH - 18);
    d.print("U1");
  }
  else if(currentMode==MODE_TEST){

    float conf = (pred==1)?p:(1-p);
    const char* phrase;
    if(conf>=0.90f)      phrase="Je suis TRES sur.";
    else if(conf>=0.75f) phrase="Assez sur.";
    else if(conf>=0.60f) phrase="Probablement.";
    else                 phrase="Pas sur.";

    // Décision
    if(conf>=DECISION_THRESHOLD){
      d.setTextColor(TFT_GREEN,TFT_BLACK);
      d.printf("TEST OK -> user %d (%.2f)", pred, conf);
    } else {
      d.setTextColor(TFT_RED,TFT_BLACK);
      d.printf("TEST REJET (%.2f)", conf);
    }

    d.setCursor(0,infoY+52);
    d.setTextColor(TFT_WHITE,TFT_BLACK);
    d.print(phrase);

    // Barre de confiance
    int barY  = infoY + INFO_H - 35;   // un peu plus haut
    int barH  = 12;
    int barW  = (int)(conf * sw);

    d.fillRect(0, barY, sw, barH, TFT_BLACK);  // fond
    uint16_t col = (conf >= DECISION_THRESHOLD) ? TFT_GREEN : TFT_RED;
    d.fillRect(0, barY, barW, barH, col);
  }
  else { // MODE_TRAIN
    d.setTextColor(TFT_CYAN,TFT_BLACK);
    d.printf("TRAIN: pred=%d  p=%.2f",pred,p);
  }

  // Logs série
  Serial.println("===== Essai =====");
  Serial.print("Mode: "); Serial.println(modeName(currentMode));
  Serial.print("User courant: "); Serial.println(currentUser);
  Serial.print("Code: "); Serial.println(currentCode);
  Serial.printf("t1=%.3f t2=%.3f t3=%.3f T=%.3f\n",t1,t2,t3,T);
  Serial.printf("p(user1)=%.3f pred=%d\n",p,pred);

  currentCode="";
  digitIndex=0;
  updateHeader();
}

// ---------- MODES & TOUCH ----------
void cycleMode(){
  if(currentMode==MODE_TRAIN) currentMode=MODE_EVAL;
  else if(currentMode==MODE_EVAL) currentMode=MODE_TEST;
  else currentMode=MODE_TRAIN;

  if(currentMode==MODE_EVAL){
    evalTotal   = 0;
    evalCorrect = 0;
  }

  updateHeader();
}

void handleTouch(int16_t tx,int16_t ty){
  auto &d=M5.Display;
  int sw=d.width(), sh=d.height();

  // Haut : changer USER (sauf en TEST)
  if(ty<HEADER_H){
    if(currentMode!=MODE_TEST){
      currentUser=(currentUser==0)?1:0;
      updateHeader();
    }
    return;
  }

  // Bas : barre contrôle
  if(ty>sh-CONTROL_H){
    if(tx<sw/2){
      // Effacer
      currentCode="";
      digitIndex=0;
      updateHeader();

      int infoY = sh - CONTROL_H - INFO_H;
      d.fillRect(0,infoY,sw,INFO_H,TFT_BLACK);
      d.setTextColor(TFT_WHITE,TFT_BLACK);
      d.setTextSize(2);
      d.setCursor(10,infoY + INFO_H/2 - 8);
      d.print("Code efface");
    } else {
      // Changer de mode
      cycleMode();
    }
    return;
  }

  // Clavier
  for(int i=0;i<12;i++){
    if(keys[i].label==' ') continue;
    int x=keys[i].x, y=keys[i].y, w=keys[i].w, h=keys[i].h;

    if(tx>=x && tx<=x+w && ty>=y && ty<=y+h){
      if(currentCode.length()>=4) return;

      currentCode+=keys[i].label;
      timeStamps[digitIndex]=millis();
      digitIndex++;

      updateHeader();

      if(digitIndex==4) processCode();
      return;
    }
  }
}

// ---------- SETUP / LOOP ----------
void setup(){
  auto cfg=M5.config();
  M5.begin(cfg);
  M5.Display.setRotation(0);
  M5.Display.setTextFont(2);

  Serial.begin(115200);
  delay(500);

  drawKeyboard();

  Serial.println("M5 - Authentification comportementale");
  Serial.println("Modes : TRAIN / EVAL / TEST");
  Serial.println("Haut : changer USER (TRAIN/EVAL).");
  Serial.println("Bas G : effacer (affiche 'Code efface').");
  Serial.println("Bas D : changer de mode.");
}

void loop(){
  M5.update();
  auto t=M5.Touch.getDetail();
  if(t.wasPressed() && t.x>=0 && t.y>=0){
    handleTouch(t.x,t.y);
  }
  delay(10);
}
