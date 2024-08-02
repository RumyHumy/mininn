#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// #Mat2d
typedef struct Mat2d {
	double **dat;
	size_t rows;
	size_t cols;
} Mat2d;

#define MATFOR_ROWS(mat,r) \
for (size_t r=0;r<mat->rows;r++)
#define MATFOR_COLS(mat,c) \
for (size_t c=0;c<mat->cols;c++)
//#define MATFOR_DEPTH(mat,c) \
//for (size_t d=0;d<mat->depth;d++)

#define MATFOR2D(mat,r,c) \
MATFOR_COLS(mat,r) \
MATFOR_ROWS(mat,c)

//#define MATFOR3D(mat,r,c) \
//MATFOR_COLS(mat,r) \
//MATFOR_ROWS(mat,c)
//MATFOR_DEPTH(mat,c)

Mat2d* Mat2dAlloc(size_t rows,size_t cols) {
	Mat2d *mat = malloc(sizeof(*mat));
	if(!mat) return NULL;
	*mat=(Mat2d){.dat=NULL,.rows=rows,.cols=cols};
	mat->dat = calloc(rows,sizeof(*mat->dat));
	if(!mat->dat){
		free(mat);
		return NULL;
	}
	MATFOR_ROWS(mat, r){
		mat->dat[r]=calloc(cols,sizeof(*mat->dat));
		if(!mat->dat[r]){
			for(size_t i=0; i<r; i++)
				free(mat->dat[r]);
			free(mat->dat);
			free(mat);
		}
	}
	return mat;
}

void Mat2dFree(Mat2d *mat){
	if (!mat) return;
	MATFOR_ROWS(mat, r)
		free(mat->dat[r]);
	free(mat->dat);
	free(mat);
}

void Mat2dLog(Mat2d *mat){
	printf("%zux%zu\n", mat->rows, mat->cols);
	MATFOR_ROWS(mat,r){
		MATFOR_COLS(mat,c)
			printf("%lf\t",mat->dat[r][c]);
		printf("\n");
	}
}
// #MSE
double MSE(Mat2d *exp,Mat2d *act){
	double **expd=exp->dat;
	double **actd=act->dat;
	double sum=0.0;
	MATFOR_COLS(exp,c)
		sum+=(actd[c][0]-expd[c][0])/exp->cols;
	return sum;
}

void MSEDeriv(
	Mat2d *exp,Mat2d *act,Mat2d *res)
{
	double **expd=exp->dat;
	double **actd=act->dat;
	double **resd=res->dat;
	MATFOR_ROWS(exp,r)
		resd[r][0]=
			2*(actd[r][0]-expd[r][0])/exp->rows;
}
// #Activations
typedef struct NetActiv{
	double (*func     )(double);
	double (*funcDeriv)(double);
} NetActiv;

// NULL (for the last layer)
#define ACTIV_NULL (NetActiv){NULL,NULL}

// Linear
double ActivLinear(double val){return val;}
double ActivLinearDeriv(double val){return 1.0;}
#define ACTIV_LINEAR \
(NetActiv){ActivLinear,ActivLinearDeriv}

// Sigmoid
double ActivSigmoid(double val){
	return 1.0/(1.0+exp(-val));
}
double ActivSigmoidDeriv(double val){
	double sigm=ActivSigmoid(val);
	return sigm*(1.0-sigm);
}
#define ACTIV_SIGMOID \
(NetActiv){ActivSigmoid,ActivSigmoidDeriv}

// #Layer
typedef struct NetLayer{
	size_t size;
	Mat2d  *ws;     // weights (next x curr)
	Mat2d  *bs;     // biases  (next x 1)
	Mat2d  *din;    // dense input (curr x 1)
	Mat2d  *ain;    // activ input (next x 1)
	NetActiv activ; // activ functions
} NetLayer;

NetLayer* NetLayerAlloc(
	size_t size,
	size_t nsize, // =0 when last layer
	NetActiv activ)
{
	NetLayer *layer = malloc(sizeof(*layer));
	if (!layer) return NULL;
	*layer = (NetLayer){
		.size  = size,
		.ws    = NULL,
		.bs    = NULL,
		.din   = NULL,
		.ain   = NULL,
		.activ = activ};
	layer->din=Mat2dAlloc(size,1);
	if (!layer->din){
		free(layer);
	}
	layer->ain=Mat2dAlloc(nsize,1);
	if (!layer->ain){
		Mat2dFree(layer->din);
		free(layer);
	}
	if (nsize!=0){
		layer->ws=Mat2dAlloc(nsize,size);
		if (!layer->ws){
			Mat2dFree(layer->din);
			Mat2dFree(layer->ain);
			free(layer);
			return NULL;
		}
		layer->bs=Mat2dAlloc(nsize,1);
		if (!layer->bs){
			Mat2dFree(layer->ws);
			Mat2dFree(layer->din);
			Mat2dFree(layer->ain);
			free(layer);
			return NULL;
		}
	}
	return layer;
}

void NetLayerFree(NetLayer* layer){
	Mat2dFree(layer->ws);
	Mat2dFree(layer->bs);
	Mat2dFree(layer->din);
	Mat2dFree(layer->ain);
	free(layer);
}

void NetLayerForw(
	NetLayer *layer,Mat2d *in,Mat2d *out)
{
	double **outd=out->dat;
	double **ind=in->dat;
	double **dind=layer->din->dat;
	double **aind=layer->ain->dat;
	double **wsd=layer->ws->dat;
	double **bsd=layer->bs->dat;
	// Dense: Y = W dot X + B
	MATFOR_ROWS(in,r)
		dind[r][0]=ind[r][0];
	MATFOR_ROWS(layer->ws,r){
		outd[r][0]=bsd[r][0];
		MATFOR_COLS(layer->ws,k)
			outd[r][0]+=wsd[r][k]*ind[k][0];
	}
	// Activation: Y' = sigmoid(Y)
	MATFOR_ROWS(out,r){
		outd[r][0]=layer->activ.func(outd[r][0]);
		aind[r][0]=outd[r][0];
	}
}

void NetLayerBackw(
		NetLayer *layer,
		Mat2d *gradIn,
		Mat2d *gradOut,
		double lrate)
{
	double **gradInd=gradIn->dat;
	double **gradOutd=gradOut->dat;
	double **dind=layer->din->dat;
	double **aind=layer->ain->dat;
	double **wsd=layer->ws->dat;
	double **bsd=layer->bs->dat;
	// Activation
	// dE/dX = dE/dY mul f'(X)
	MATFOR2D(gradIn,r,c)
		gradInd[r][c]*=
			layer->activ.funcDeriv(aind[r][c]);
	// Dense
	// W' = W - lrate * (dY/dW x X^T)(.)
	MATFOR_ROWS(layer->ws,r){
		MATFOR_COLS(layer->ws,k)
			wsd[r][k]-=lrate*gradInd[r][0]*dind[k][0];
		bsd[r][0]-=lrate*gradInd[r][0];
	}
	// dE/dX = W^T * dE/dY
	MATFOR_COLS(layer->ws,c){
		gradOutd[c][0]=0.0;
		MATFOR_ROWS(layer->ws,k)
			gradOutd[c][0]+=wsd[k][c]*gradInd[k][0];
	}
}

// #Net
typedef struct Net{
	size_t     lcount; // layers count
	NetLayer **layers; // layers data
} Net;

Net* NetAlloc(
	size_t lcount,
	size_t ncounts[],
	NetActiv activs[])
{
	Net* net=malloc(sizeof(*net));
	if (!net) return NULL;
	*net=(Net){.lcount=lcount,.layers=NULL};
	net->layers=
		malloc(lcount*sizeof(*net->layers));
	if (!net->layers){
		free(net);
		return NULL;
	}
	for (int l=0; l<lcount; l++){
		net->layers[l]=NetLayerAlloc(
			ncounts[l],
			(l<lcount-1?ncounts[l+1]:0),
			(l<lcount-1?activs[l]:ACTIV_NULL));
		if (!net->layers[l]){
			for (int i=0; i<l; i++)
				NetLayerFree(net->layers[i]);
			free(net->layers);
			free(net);
			return NULL;
		}
	}
	return net;
}

void NetFree(Net* net){
	if (!net) return;
	for (int l=0; l<net->lcount; l++)
		NetLayerFree(net->layers[l]);
	free(net->layers);
	free(net);
}

void NetSimpleTrain(
	Net *net,
	size_t tcount,
	Mat2d *tIn[],
	Mat2d *tOut[],
	size_t epochs)
{
	//Mat2d *out=Mat2dAlloc(tOut[0]->rows, 1);
	//for (int e=0; e<epochs; e++){
	//	int i=rand()%tcount;
	//	Mat2d *in=tIn[i];
	//	Mat2d *eOut=tOut[i];
	//	for (int l=0; l<net->lcount; l++){
	//		NetLayerForw(
	//			net->layers[l],in,out);
	//	}
	//	MSEDeriv(eOut,out,grad);
	//}
	//Mat2dFree(out);
}

// #Unit tests
#define FCOMP(f1,f2,p) \
roundf(f1*pow(10,p))==roundf(f2*pow(10,p))

char UTMat2dAlloc(){
	char flag=0;
	Mat2d *mat=Mat2dAlloc(2, 3);
	if (!mat) goto cleanup;
	//Mat2dLog(mat);
	double **m = mat->dat;
	flag=m[0][0]==0.0&&m[1][2]==0.0;
cleanup:
	Mat2dFree(mat);
	return flag;
}

char UTNetActiv()
{
	char flag=1;
	flag&=FCOMP(ACTIV_LINEAR.func(1.5),1.5,1);
	flag&=FCOMP(ACTIV_LINEAR.funcDeriv(9.9),1.0,1);
	flag&=FCOMP(ACTIV_SIGMOID.func(0),0.5,1);
	flag&=FCOMP(ACTIV_SIGMOID.funcDeriv(0),0.25,2);
	return flag;
}

char UTNetLayer(){
	char flag=0;
	NetLayer *layer=NetLayerAlloc(
			2, 3, ACTIV_SIGMOID);
	if (!layer) return 0;
	flag=layer->size==2
		&& layer->ws->rows==3
		&& layer->ws->cols==2
		&& layer->bs->rows==3
		&& layer->bs->cols==1
		&& layer->activ.func==ActivSigmoid
		&& layer->activ.funcDeriv==ActivSigmoidDeriv;
cleanup:
	NetLayerFree(layer);
	return flag;
}

char UTNetLayerForw(){
	char flag=0;
	NetLayer *layer=NetLayerAlloc(
		2, 3, ACTIV_SIGMOID);
	if (!layer) goto cleanup;
	Mat2d *in=Mat2dAlloc(2, 1);
	if (!in) goto cleanup;
	Mat2d *out=Mat2dAlloc(3, 1);
	if (!out) goto cleanup;
	//      0.0 -0.5      +0.5
	// W = +0.5 +1.0; B = -1.5
	//     -1.0  0.0      +0.5
	double **wsd=layer->ws->dat;
	wsd[0][0]= 0.0; wsd[0][1]=-0.5;
	wsd[1][0]=+0.5; wsd[1][1]=+1.0;
	wsd[2][0]=-1.0; wsd[2][1]= 0.0;
	double **bsd=layer->bs->dat;
	bsd[0][0]=+0.5;
	bsd[1][0]=-1.5;
	bsd[2][0]=-0.5;
	//     +2.0      0.731...
	// X = -1.0; Y = 0.182...
	//               0.076...
	double **ind=in->dat;
	ind[0][0]=+2.0;
	ind[1][0]=-1.0;
	NetLayerForw(layer,in,out);
	flag=FCOMP(out->dat[0][0],0.731,3)
		&& FCOMP(out->dat[1][0],0.182,3)
		&& FCOMP(out->dat[2][0],0.076,3);
cleanup:
	NetLayerFree(layer);
	return flag;
}

char UTNetLayerBackw(){
	char flag=0;
	NetLayer *layer=NetLayerAlloc(
		2, 3, ACTIV_SIGMOID);
	if (!layer) goto cleanup;
	Mat2d *in=Mat2dAlloc(2, 1);
	if (!in) goto cleanup;
	Mat2d *out=Mat2dAlloc(3, 1);
	if (!out) goto cleanup;
	Mat2d *eout=Mat2dAlloc(3, 1);
	if (!eout) goto cleanup;
	Mat2d *grad=Mat2dAlloc(3, 1);
	if (!grad) goto cleanup;
	Mat2d *gradOut=Mat2dAlloc(2, 1);
	if (!gradOut) goto cleanup;
	//     +2.0       0.666...
	// X = -1.0; Y* = 0.077...
	//                0.500...
	double **ind=in->dat;
	ind[0][0]=+2.0;
	ind[1][0]=-1.0;
	double **eoutd=eout->dat;
	eoutd[0][0]=0.666;
	eoutd[1][0]=0.077;
	eoutd[2][0]=0.500;
	for (int i=0; i<10000; i++){
		NetLayerForw(layer,in,out);
		MSEDeriv(eout,out,grad);
		NetLayerBackw(layer,grad,gradOut,0.01);
	}
	double **outd=out->dat;
	flag=FCOMP(out->dat[0][0],0.666,3)
		&& FCOMP(out->dat[1][0],0.077,3)
		&& FCOMP(out->dat[2][0],0.500,3);
cleanup:
	NetLayerFree(layer);
	Mat2dFree(in);
	Mat2dFree(out);
	Mat2dFree(eout);
	Mat2dFree(grad);
	Mat2dFree(gradOut);
	return flag;
}

char UTNet(){
	char flag=0;
	Net *net=NetAlloc(
		4,
		(size_t[]){2,3,4,2},
		(NetActiv[]){
			ACTIV_SIGMOID,
			ACTIV_SIGMOID,
			ACTIV_SIGMOID});
	if (!net) goto cleanup;
	flag=1
		&& net->layers[0]->size==2
		&& net->layers[1]->size==3
		&& net->layers[2]->size==4
		&& net->layers[3]->size==2;
cleanup:
	NetFree(net);
	return flag;
}

char UTNetSimpleTrain(){
	char flag=0;
	Net *net=NetAlloc(
		3,
		(size_t[]){2,3,1},
		(NetActiv[]){
			ACTIV_SIGMOID,
			ACTIV_SIGMOID,
			ACTIV_SIGMOID});
	if (!net) goto cleanup;

	//NetSimpleTrain(net,4,tin,tout,10000);
cleanup:
	NetFree(net);
	return flag;
}

void UnitTests(){
	printf("UNIT TESTS:\n");
	printf("Mat Alloc:            %d\n",
		UTMat2dAlloc());
	printf("Net Activ:            %d\n",
		UTNetActiv());
	printf("Net Layer:            %d\n",
		UTNetLayer());
	printf("Net Layer Forw:       %d\n",
		UTNetLayerForw());
	printf("Net Layer Backw:      %d\n",
		UTNetLayerBackw());
	printf("Net Net:              %d\n",
		UTNet());
	printf("Net Net Simple Train: %d\n",
		UTNetSimpleTrain());
}

int main()
{
	UnitTests();

	return 0;
}
