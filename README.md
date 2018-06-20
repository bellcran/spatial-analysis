# 공간분석 주요 용어

### scale, resolution

**scale**

    - 실제 거리에 대한 (종이)지도상의 거리 비율
    - 규모, 척도, 축척

- 지도 "A"에서 1cm의 거리가 실제 세계에서 100m를 나타내는 경우, 1 / 10,000
- 지도 "B"에서 1cm가 실제 세계에서 10km를 나타낼 경우 해당 지도의 scale 은 1 / 1,000,000

  * "A"는 "B"에 비해 상대적으로 큰 배율 (고해상도)
  * "A"와 "B" 지도 크기가 동일하다면 지도 "B"는 훨씬 더 큰 영역을 표현 (훨씬 더 큰 "공간 범위"를 가짐)

----------------

**resolution**

    - 래스터데이터 해상도 : 셀 크기
    - 벡터데이터 해상도 : 라인 또는 폴리곤의 노드 (좌표 쌍) 간 평균 거리
    - 포인트데이터는 해상도가 없다.

- scale, resolution 중요한 이유
  1. 길이와 크기 추정에 영향을 준다.
  2. 관심변수 간 관계에 대한 이해에 영향을 준다.

- 데이터 수집 고려 사항
  1. 데이터는 가능한 가장 높은 공간적 (시간적) 해상도를 가지도록 한다.
  2. 데이터를 저해상도로 집계 할 수는 있지만 데이터를 고해상도로 분류, 축소하는 것은 쉽지 않다.

---------

### aggregation(집계), zonation (구역설정, 영역분할)

- 공간데이터에서 데이터를 집계하는 데 사용되는 영역은 임의적이다.

- 수정 가능한 단위 문제 (Modifiable Areal Unit Problem; MAUP)
  - 영역의 경계가 그려지는 방식 (크기, 모양, 크기)이 우리가 보는 패턴과 데이터 분석 결과에 영향을 주는 것
- 생태학적 추론 (Ecological Inference)
  - 집계된 데이터를 분석하는 것


**(ex) 구역설정과 집계효과 설명 예제**

1. 소득분포 자료 생성
```
set.seed(0)
xy <- cbind(x=runif(1000, 0, 100), y=runif(1000, 0, 100))
income <- (runif(1000) * abs((xy[,1] - 50) * (xy[,2] - 50))) / 500
```

2. 일반분석
```
par(mfrow=c(1,3), las=1)
plot(sort(income), col=rev(terrain.colors(1000)), pch=20, cex=.75, ylab='income')
hist(income, main='', col=rev(terrain.colors(10)),  xlim=c(0,5), breaks=seq(0,5,0.5))
plot(xy, xlim=c(0,100), ylim=c(0,100), cex=income, col=rev(terrain.colors(50))[10*(income+1)])
```
   - 처음 두 개의 그림은 많은 빈민층과 소수의 부자가 있음을 보여준다. 
   - 세번째 그림은 부자와 가난한 자들이 사는 곳에 명확한 공간적 패턴이 있다는 것을 보여준다.

3. 공간분석

  - 각기 다른 지역 즉, 직사각형의 래스터 셀을 만들고 각 지역의 평균 소득을 계산한다.
```
library(raster)
r1 <- raster(ncol=1, nrow=4, xmn=0, xmx=100, ymn=0, ymx=100, crs=NA)
r1 <- rasterize(xy, r1, income, mean)

r2 <- raster(ncol=4, nrow=1, xmn=0, xmx=100, ymn=0, ymx=100, crs=NA)
r2 <- rasterize(xy, r2, income, mean)

r3 <- raster(ncol=2, nrow=2, xmn=0, xmx=100, ymn=0, ymx=100, crs=NA)
r3 <- rasterize(xy, r3, income, mean)

r4 <- raster(ncol=3, nrow=3, xmn=0, xmx=100, ymn=0, ymx=100, crs=NA)
r4 <- rasterize(xy, r4, income, mean)

r5 <- raster(ncol=5, nrow=5, xmn=0, xmx=100, ymn=0, ymx=100, crs=NA)
r5 <- rasterize(xy, r5, income, mean)

r6 <- raster(ncol=10, nrow=10, xmn=0, xmx=100, ymn=0, ymx=100, crs=NA)
r6 <- rasterize(xy, r6, income, mean)

par(mfrow=c(2,3), las=1)
plot(r1); plot(r2); plot(r3); plot(r4); plot(r5); plot(r6)
```
- 영역이 작을수록 실제 패턴을 잘 포착한다

```
par(mfrow=c(1,3), las=1)
hist(r4, main='', col=rev(terrain.colors(10)), xlim=c(0,5), breaks=seq(0, 5, 0.5))
hist(r5, main='', col=rev(terrain.colors(10)), xlim=c(0,5), breaks=seq(0, 5, 0.5))plo
hist(r6, main='', col=rev(terrain.colors(10)), xlim=c(0,5), breaks=seq(0, 5, 0.5))
```
- 히스토그램은 전체 소득 분포를 포착하지 못한다.

----------
### 거리와 인접성
    - 공간분석에서 거리는 국경, 산 또는 다른 장벽 등을 포함한다.
    - A와 B 사이의 거리는 비대칭이다. 즉, A에서 B까지의 거리가 B에서 A까지의 거리와 같지 않다.
    - 분석에서 거리는 "거리 행렬"로 표현된다.
    - 거리가 대칭이면 매트릭스의 절반만 채워진다.

**(ex) 거리행렬 예제**

1. 자료 생성
```
A <- c(40, 43)
B <- c(101, 1)
C <- c(111, 54)
D <- c(104, 65)
E <- c(60, 22)
F <- c(20, 2)
pts <- rbind(A, B, C, D, E, F)
pts
```
2. 그래프
```
par(mfrow=c(1,1))
plot(pts, xlim=c(0,120), ylim=c(0,120), pch=20, cex=2, col='red', xlab='X', ylab='Y', las=1)
text(pts+5, LETTERS[1:6])
```
3. 거리행렬 생성
```
dis <- dist(pts)
dis
```
4. 정규행렬 변환
```
D <- as.matrix(dis)
round(D)
```
- 거리행렬은 비지리적 데이터 분석에 사용

**(ex) 경위도 좌표 거리 예제**

- pts의 값이 위경도 좌표라고 가정
```
library(raster)
gdis <- pointDistance(pts, lonlat=TRUE)
gdis
```
----------
### 공간 영향 (Spatial influence)
    - 공간분석을 위해 지리적 객체 간의 공간 영향을 측정해야 한다.
    - 공간영향은 인접성과 거리함수로 표현된다. 
    - 인접성은 인접행렬로 표현된다.
    - 거리함수 혹은 역거리함수는 공간 가중치 행렬로 표현된다.

**인접성**

포인트데이터를 분석하는 일반적인 방법

**(ex) 인접행렬 예제**

1. 포인트가 서로 50도 이내인 경우 "인접한" 포인트로 정의
```
a <- D < 50
a
```
2. 인접행렬 생성
```
diag(a) <- NA
Adj50 <- a * 1
Adj50
```
**(ex)"두 개의 가장 가까운 이웃" 인접 행렬 생성**
```
cols <- apply(D, 1, order)
cols <- t(cols)
cols <- cols[,2:3]  # 열 번호를 얻는다.

rowcols <- cbind(rep(1:6, each=2), as.vector(t(cols))) # 행-열 쌍을 만든다.
head(rowcols) 

Adj50
Ak3 <- Adj50 * 0
Ak3[rowcols] <- 1  # 이 쌍들을 색인으로 사용하여 행렬 Ak3의 값을 변경한다.
Ak3
```

**가중값 행렬**

공간 영향을 이진 값 (인접 또는 비공유)으로 표현하는 대신 연속 값으로 표현하는 경우
(역)거리를 사용한다. (멀리 떨어 뜨리면 값이 낮아짐).

**(ex) 공간 가중치 행렬 예제**
```
W <- 1 / D  # (역)거리 계산
round(W, 4)

W[!is.finite(W)] <- NA # Inf 값을 NA로 변경하여 제거

rtot <- rowSums(W, na.rm=TRUE) #  행 합계를 계산
# this is equivalent to
# rtot <- apply(W, 1, sum, na.rm=TRUE)
rtot

W <- W / rtot # 행을 합계로 나눈다.
rowSums(W, na.rm=TRUE)  # 행 합계가 1인지 확인
colSums(W, na.rm=TRUE) # 열의 개별 값이 1을 넘지 않는다.
```

**(ex) 폴리곤 공간영향 예제**
1. 데이터 불려오기
```
library(raster)
p <- shapefile(system.file("external/lux.shp", package="raster"))
```
2. 인접한 폴리곤을 찾기 위해 spdep 패키지 사용
```
library(spdep) 
```
3. 이웃 목록 생성
```
wr <- poly2nb(p, row.names=p$ID_2, queen=FALSE)
wr
```
4. 이웃 행렬 생성
```
wm <- nb2mat(wr, style='B', zero.policy = TRUE)
dim(wm)
## [1] 12 12
```
5. wr, wm 확인
```
wr[1:6]
wm[1:6,1:11]
```

6. 각 영역에 대한 이웃 수를 계산
```
i <- rowSums(wm)
i
```
7. 백분율로 표시합니다.
```
round(100 * table(i) / length(i), 1)
```
8. 폴리곤 사이의 링크를 플롯
```
par(mai=c(0,0,0,0))
plot(p, col='gray', border='blue')
xy <- coordinates(p)
plot(wr, xy, col='red', lwd=2, add=TRUE)
```

9. 공간 영향 계산
```
## Distance based:
wd10 <- dnearneigh(xy, 0, 10)
wd25 <- dnearneigh(xy, 0, 25, longlat=TRUE)

## Nearest neighbors:
k3 <- knn2nb(knearneigh(xy, k=3, RANN=FALSE))
k6 <- knn2nb(knearneigh(xy, k=6, RANN=FALSE))

## Lag-two Rook:
wr2 <- wr
for (i in 1:length(wr)) {
    lag1 <- wr[[i]]
    lag2 <- wr[lag1]
    lag2 <- sort(unique(unlist(lag2)))
    lag2 <- lag2[!(lag2 %in% c(wr[[i]], i))]
    wr2[[i]] <- lag2
}

# 그래프; plotit 함수 사용
plotit <- function(nb, lab='') {
  plot(p, col='gray', border='white')
  plot(nb, xy, add=TRUE, pch=20)
  text(6.3, 50.1, paste0('(', lab, ')'), cex=1.25)
}

par(mfrow=c(2, 3), mai=c(0,0,0,0))
plotit(wr, 'adjacency')
plotit(wr2, 'lag-2 adj.')
plotit(wd10, '10 km')
plotit(wd25, '25 km')
plotit(k3, 'k=3')
plotit(k6, 'k=6')
```

# GWR (지리가중회귀)
    - 회귀 모델은 일반적으로 global regression 이다.
    - 일반화 가산 모델은 local regression의 대표적인 예이다.
    - 여기서 "로컬"은 예측값을 나타낸다.
    - 공간에서 로컬은 위치를 나타낸다.
    - 하나의 회귀 모델을 피팅하는 대신 여러 위치(가능한 한 많은)에 대해 
      여러 모델을 적용하는 것
    - 이를 "지리적으로 가중된 회귀(geographically weighted regression)"라고 한다.
    - GWR은 공간에 대한 다양한 변수의 중요도 변화를 이해할 수 있는 
      데이터 탐색 기술이다.
    

**(ex) 캘리포니아 강수량**

1. 데이터 불려오기

http://rspatial.org/analysis/data/precipitation.csv
http://rspatial.org/analysis/data/counties.rds
```
cts <- readRDS('data/counties.rds') #지도
p <- read.csv('data/precipitation.csv') #강수량
head(p)

plot(cts)
points(p[,c('LONG', 'LAT')], col='red', pch=20)
```
2. 연평균 강수량 계산
```
p$pan <- rowSums(p[,6:17])
Global regression model

m <- lm(pan ~ ALT, data=p)
m
```
3. crs를 사용하여 Spatial 객체 생성
```
alb <- CRS("+proj=aea +lat_1=34 +lat_2=40.5 +lat_0=0 +lon_0=-120 +x_0=0 +y_0=-4000000 +ellps=GRS80 +datum=NAD83 +units=m +no_defs")
sp <- p
coordinates(sp) = ~ LONG + LAT
crs(sp) <- "+proj=longlat +datum=NAD83"
spt <- spTransform(sp, alb)
ctst <- spTransform(cts, alb)
```
4. 최적의 대역폭 확보
```
ibrary( spgwr )

bw <- gwr.sel(pan ~ ALT, data=spt)
bw
```
5. 회귀계수를 추정하기 위해 점 집합을 생성
```
r <- raster(ctst, res=10000)
r <- rasterize(ctst, r)
newpts <- rasterToPoints(r)
```
6. gwr 실행
```
g <- gwr(pan ~ ALT, data=spt, bandwidth=bw, fit.points=newpts[, 1:2])
g
```
7. 결과를 래스터에 연결
```
slope <- r
intercept <- r
slope[!is.na(slope)] <- g$SDF$ALT
intercept[!is.na(intercept)] <- g$SDF$'(Intercept)'
s <- stack(intercept, slope)
names(s) <- c('intercept', 'slope')
plot(s)
```
**(ex) 캘리포니아 주택 가격**
1. 데이터 불려오기

http://rspatial.org/data/analysis/gwr.zip
```
houses <- read.csv("houses1990.csv")
dim(houses)
## [1] 20640     9
head(houses)
````
2. 공간데이터로 변환, 포인트 표시
```
library(sp)
coordinates(houses) <- ~longitude+latitude
plot(houses, cex=0.5, pch=1, axes=TRUE)
```
3. 공간데이터와 카운티 맵 CRS 일치, 표시
Now get the county boundaries and assign CRS of the houses data matches that of the counties (because they are both in longitude/latitude!).

```
library(raster)
counties <- readRDS("data/counties.rds")
crs(houses) <- crs(counties)
plot(counties, add=T)
```
4. 공간쿼리 수행 (points in polygon)
```
cnty <- over(houses, counties)
head(cnty)
```
5. 데이터 요약 (by county)
```
hd <- cbind(data.frame(houses), cnty)

# 카운티별 인구수 계산
totpop <- tapply(hd$population, hd$NAME, sum)
totpop

# 카운티별 평균소득 계산
hd$suminc <- hd$income * hd$households # 총수입
csum <- aggregate(hd[, c('suminc', 'households')], list(hd$NAME), sum) # aggregate 함수 사용 (similar to tapply)
csum$income <- 10000 * csum$suminc / csum$households # 총수입을 가구수로 나누기
csum <- csum[order(csum$income), ] # 정렬
head(csum)
tail(csum)
```
6. 회귀분석 수행
```
# 변수 추가
hd$roomhead <- hd$rooms / hd$population
hd$bedroomhead <- hd$bedrooms / hd$population
hd$hhsize <- hd$population / hd$households

# OLS
m <- glm( houseValue ~ income + houseAge + roomhead + bedroomhead + population, data=hd)
summary(m)
coefficients(m)
```
7. 지리적 가중회귀 수행
  - by county
```
hd2 <- hd [! is.na (hd $ NAME),] # 카운티 경계 벗어난 데이터 삭제

regfun <- function (x) { # 회귀계수 구하는 함수
  dat <- hd2[hd2$NAME == x, ]
  m <- glm(houseValue~income+houseAge+roomhead+bedroomhead+population, data=dat)
  coefficients(m)
}

countynames <- unique (hd2$NAME)
res <- sapply(countynames, regfun) # 모든 카운티별 회귀계수 구하기

dotchart (sort (res [ 'income',]), cex = 0.65) # 회귀계수 그래프 그리기

resdf <- data.frame(NAME=colnames(res), t(res)) # 지도에 표시하기 위한 데이터
head(resdf)

dim(counties) # 싱글카운티가 싱글폴리곤이 되도록 수정, 섬에 해당하는 카운티 제거된다.
## [1] 68  5
dcounties <- aggregate(counties, vars='NAME')
## Warning in .local(x, ...): Use argument "by" instead of deprecated argument
## "vars"
dim(dcounties)
## [1] 58  1

# SpatialPolygonsDataFrame을 data.frame과 병합
cnres <- merge(dcounties, resdf, by='NAME')
spplot(cnres, 'income')


cnres2 <- cnres # 데이터 사본
# 첫 번째 변수 (카운티 이름)를 제외한 모든 변수의 크기 조정
# 지도에 회귀계수를 표시하려면 값을 조정해야 한다.
cnres2@data = data.frame(scale(data.frame(cnres)[, -1]))
spplot(cnres2)

# 랜덤 잡음인지 공간자기상관인지 검정
library(spdep)
nb <- poly2nb(cnres)
plot(cnres)
plot(nb, coordinates(cnres), add=T, col='red')

lw <- nb2listw(nb)
moran.test(cnres$income, lw)
moran.test(cnres$roomhead, lw, na.action=na.omit)
```
  - by grid cell
```
# 그리드 셀의 모델 계산, 'Teale Albers' 투영법 사용
TA <- CRS("+proj=aea +lat_1=34 +lat_2=40.5 +lat_0=0 +lon_0=-120 +x_0=0 +y_0=-4000000
              +datum=NAD83 +units=m +no_defs +ellps=GRS80 +towgs84=0,0,0")
countiesTA <- spTransform(counties, TA)

library(raster)
r <- raster(countiesTA) # RasteLayer 생성
res(r) <- 50000  # 50 x 50 km 셀 해상도 설정
xy <- xyFromCell (r, 1 : ncell (r)) # 각 래스터 셀의 xy 좌표

housesTA <- spTransform(houses, TA) # Teale-Albers로 변환
crds <- coordinates(housesTA)

regfun2 <- function(d)  { # 새로운 회귀함수 설정
 m <- glm(houseValue~income+houseAge+roomhead+bedroomhead+population, data=d)
 coefficients(m)
}

res <- list()
for (i in 1:nrow(xy)) { # 50km 반경 내 최소 50 회 관측, 다른 셀에서 사용되는 데이터와 겹칩
    d <- sqrt((xy[i,1]-crds[,1])^2 + (xy[i,2]-crds[,2])^2)
    j <- which(d < 50000)
    if (length(j) > 49) {
        d <- hd[j,]
        res[[i]] <- regfun2(d)
    } else {
        res[[i]] <- NA
    }
}

inc <- sapply(res, function(x) x['income']) # 각 셀에 대한 소득계수

rinc <- setValues(r, inc) # RasterLayer에 값 적용
plot(rinc)
plot(countiesTA, add=T)

Moran(rinc)
## [1] 0.3271564
```

**spgwr 패키지 사용**
```
r <- raster(countiesTA) # 래스터 레이어 만들기
res(r) <- 25000 # 원하는 해상도로 설정

# CA의 내부에만 셀이 존재해야 한다.
ca <- rasterize (countiesTA, r) # NA가 아닌 좌표 추출
fitpoints <- rasterToPoints (ca) 
fitpoints <- fitpoints [, - 3] # 세 번째 칼럼 제거
#gwr.model <- ______ # 모델 구체화
col.bw <- gwr.sel(houseValue~income+houseAge+population, data=housesTA)
gwr.model <- gwr(houseValue~income+houseAge+population, data=housesTA,
                 bandwidth=col.bw, fit.points=fitpoints) # 모델 구체화

sp <- gwr.model$SDF # patialPointsDataFrame 추출
spplot(sp) # SpatialPointsDataFrame 그래프 그리기

# 값을 래스터 구조에 재연결
cells <- cellFromXY(r, fitpoints)
dd <- as.matrix(data.frame(sp))
b <- brick(r, values=FALSE, nl=nrow(dd)) # RasterBrick 객체로 변환
b[cells] <- dd
names(b) <- colnames(dd)
plot(b)
```














