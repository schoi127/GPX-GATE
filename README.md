# Project Title 

이 프로젝트는 Geant4 기반의 입자 시뮬레이션 중 하나인 GATE (Geant4 Application for Tomographic Emission) 툴킷을 활용한
지면 투과 후방산란 엑스선 (GPX) 영상 시뮬레이션을 위하여 지능형부품센서연구실에서 개발한 SW입니다.
주요 기능 및 특징은 다음과 같습니다. 
<br>
<br>
<br>

## Table of Contents
- [Environment & Prerequisites](#environment-and-prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Authors](#authors)

<br>


## Environment and Prerequisites
- 개발언어:  ROOT 6.22.08, Geant4 10.07.p01, GATE 9.1
- OS:  Ubuntu 20.04

<br>


## Installation

- GATE 설치를 위해 OpenGATECollaboration 홈페이지를 참조합니다.
- 소스코드를 다운로드 받아서 실행합니다. 

```bash
git clone https://github.com/schoi127/GATE-GTX
```

<br>


## Usage 
- Step 별로 차례대로 수행합니다.
- Step 1) Electron to Fan-beam X-ray phase space
- Step 2) GPX system run session #1: 콜리메이터 유무에 따른 후방산란 엑스선 시뮬레이션 
- Step 2) GPX system run session #2: 흙 밀도 고정, 투과 깊이에 따른 물질별 후방산란 엑스선 시뮬레이션
- Step 3) GPX system run session #3: 투과 깊이 고정, 흙 밀도에 따른 물질별 후방산란 엑스선 시뮬레이션
- Step 4) GPX system run session #4: 흙 밀도 고정, 0cm 표면, 엑스선 입사 에너지에 따른 물질별 후방산란율 계산  

<br>
<br>
<br>
<br>
<br>


## Authors
* 최성훈 &nbsp;&nbsp;&nbsp;  schoi@etri.re.kr   

<br>
<br>
<br>


## Version
- 1.0
<br>
<br>

