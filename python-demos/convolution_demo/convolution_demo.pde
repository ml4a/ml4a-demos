
int w = 20;
int h = 20;
int iw = 24;
int ih = 24;
int fw = 5;
int fh = 5;
/*
int iz = -400;
int fz = 0;
int cz = 400;
*/

int iz = 0;
int fz = 200;
int cz = 800;


int cw, ch;
int fx, fy;
int mx, my;
  
void setup() {
  size(1280, 800, P3D);
  fx = 0;
  fy = 0;
  cw = iw - fw + 1;
  ch = ih - fh + 1;
  mx = (int) (0.5 * w * (iw - cw));
  my = (int) (0.5 * h * (ih - ch));
}

void drawImage() {
  pushMatrix();

  stroke(0);
  for (int i=0; i<=iw; i++) {
    int x = i * w;
    int y1 = 0;
    int y2 = ih * h;
    line(x, y1, iz, x, y2, iz); 
  }
  for (int i=0; i<=ih; i++) {
    int y = i * h;
    int x1 = 0;
    int x2 = iw * w;
    line(x1, y, iz, x2, y, iz); 
  }
  
  popMatrix();
}

void drawFilter() {
  pushMatrix();
  
  stroke(0);
  for (int i=0; i<=fw; i++) {
    int x = (fx + i) * w;
    int y1 = fy * h;
    int y2 = (fy + fh) * h;
    line(x, y1, fz, x, y2, fz); 
  }
  for (int i=0; i<=fh; i++) {
    int y = (fy + i) * h;
    int x1 = fx * w;
    int x2 = (fx + fw) * w;
    line(x1, y, fz, x2, y, fz); 
  }
  
  popMatrix();
}

void drawConv() {
  pushMatrix();
  
  
  stroke(0);
  for (int i=0; i<=cw; i++) {
    int x = i * w + mx;
    int y1 = my;
    int y2 = my + ch * h;
    line(x, y1, cz, x, y2, cz); 
  }
  for (int i=0; i<=ch; i++) {
    int y = i * h + my;
    int x1 = mx;
    int x2 = mx + cw * w;
    line(x1, y, cz, x2, y, cz); 
  }
  
  popMatrix();
}

void draw() {
  background(255);
  translate(width/4, height/4);
  rotateY(mouseX/100.0);
  drawImage();
  drawFilter();
  drawConv();
  drawMapping();
  
  if (frameCount % 5 == 0) {
    next();
  }
}

void drawMapping() {
  
  int xi = fx * w;
  int yi = fy * h; 
  int zi = iz;
  
  int xf = fx * w;
  int yf = fy * h;
  int zf = fz;
  
  int xc = mx + fx * w;
  int yc = my + fy * h;
  int zc = cz;
  
  int margin_x = fw * w;
  int margin_y = fh * h;
  
  stroke(0, 100);
  line(xi, yi, zi, xf, yf, zf);
  line(xi + margin_x, yi, zi, xf + margin_x, yf, zf);
  line(xi + margin_x, yi + margin_y, zi, xf + margin_x, yf + margin_y, zf);
  line(xi, yi + margin_y, zi, xf, yf + margin_y, zf);
  
  line(xc, yc, zc, xf, yf, zf);
  line(xc + w, yc, zc, xf + margin_x, yf, zf);
  line(xc + w, yc + h, zc, xf + margin_x, yf + margin_y, zf);
  line(xc, yc + h, zc, xf, yf + margin_y, zf);
  
}

void next() {
  fx += 1;
  if (fx > iw - fw) {
    fx = 0;
    fy += 1;
    if (fy > ih - fh) {
      fy = 0;
    }
  }
}

