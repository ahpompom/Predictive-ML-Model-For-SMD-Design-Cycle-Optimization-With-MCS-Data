# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 15:39:49 2022
to perform geometrical dimensioning and tolerancing simulation for Tiber 8mm  
module (with discrete core). 

@author: chuako
"""

import numpy as np
import random
from shapely.geometry import Polygon, Point
from shapely.affinity import rotate
import matplotlib.pyplot as plt 
import time
import pandas as pd

###############################################################################
###################### User defined functions #################################
##############################################################################

def Normal(x, dx, Nsigma, N):
    ''' this function return a data set of x with tolerance dx. Sample size = N.
    dx = tolerance of the dimension (based on 3sigma). If dx = 0, this function
    return a array of x with sample size of N.
    '''
    if dx == 0:
        return np.full(N, x)
    else: return np.random.normal(x, dx/Nsigma, N)

def Rectangle(x_center,y_center,length, height, beta, x_rot, y_rot):
    ''' this function return a rectangle geometry of baseplate, with input of 
    the center coordinates, length and height of the rectangle's geometry.
    With consideration of tolerance, one can include the tolerance separately in
    the rectangle's center coordinate, length and height. A rotational can be 
    input through rotate arguement (beta, in unit of degrees). N = sample size.
    '''
    
    rec = Polygon( [(x_center - 0.5*length, y_center - 0.5*height), 
                   (x_center + 0.5*length, y_center - 0.5*height), 
                   (x_center + 0.5*length, y_center + 0.5*height), 
                   (x_center - 0.5*length, y_center + 0.5*height)] )
    return rotate(rec, beta, (x_rot, y_rot) )

def Tubecore(x_center, y_center, width, length, fillet, beta, x_rot, y_rot):
    ''' this function return a rectangle geometry of baseplate, with input of 
    the center coordinates, length and height of the rectangle's geometry.
    With consideration of tolerance, one can include the tolerance separately in
    the rectangle's center coordinate, length and height. A rotational can be 
    input through rotate arguement (beta, in unit of degrees). N = sample size.
    x_rot, y_rot are the rotation axis.     '''
          
    tubecore = Polygon( [
        (x_center - 0.5*width + fillet, y_center - 0.5*length), 
        (x_center + 0.5*width - fillet, y_center - 0.5*length),
        (x_center + 0.5*width, y_center - 0.5*length + fillet),
        (x_center + 0.5*width, y_center + 0.5*length - fillet),
        (x_center + 0.5*width - fillet, y_center + 0.5*length),
        (x_center - 0.5*width + fillet, y_center + 0.5*length),
        (x_center - 0.5*width, y_center + 0.5*length - fillet),
        (x_center - 0.5*width, y_center - 0.5*length + fillet) ] )
    
    return rotate(tubecore, beta, (x_rot, y_rot))


def beta(W_Base, L_Base, X_base, Y_base, W_Denali, L_Denali, X_Denali, Y_Denali ):
    ''' this function return how much the baseplate can retate due to gape between
    baseplate and Denali.HDA top solder mask. The returned value reflect the max
    rotational angle of the baseplate'''
       
    gap_xr = 0.5*(W_Denali-W_Base) + (X_base - X_Denali ) # the last term is theoretical position of base
    gap_xl = 0.5*(W_Denali-W_Base) - (X_base - X_Denali )
    gap_ytop = 0.5*(L_Denali-L_Base) + (Y_base - Y_Denali )
    gap_ybot = 0.5*(L_Denali-L_Base) - (Y_base - Y_Denali )
    gap_xmin = np.minimum(gap_xr, gap_xl)
    gap_ymin = np.minimum(gap_ytop, gap_ybot)
    
    #print('gap_xmin: ', gap_xmin)
    #print('gap_ymin: ', gap_ymin)
    
    return np.degrees( np.sqrt( (gap_xmin**2 + gap_ymin**2) / (0.25*W_Base**2 + 0.25*L_Base**2)) )     # unit = degrees, randomly assign direction of rotation



def RAND(N):
    RAND = []
    for l in np.arange(N):
        rand = random.choice( [-1,1])
        RAND = np.append(RAND, rand)
    return RAND

#def Conflict(Polygon, Line):
#    return Polygon.intersects(Line)  

def Conflict(object1, object2):
    return object1.overlaps(object2)  

def RadiusRand(radius_difference):
    RadiusX = np.array([])
    RadiusY = np.array([])
    Rand = np.random.choice( [-1,1],1)
    alpha = np.random.normal(0, 0.5*np.pi, 1)
    Radius = np.random.normal(0, radius_difference, 1)
    RadiusX = np.append(RadiusX, Rand*Radius*np.cos(alpha))
    RadiusY = np.append(RadiusY, Rand*Radius*np.sin(alpha))
    return RadiusX, RadiusY

def BaseRotation(Base, DenaliOpening, resolution):
    Delta_Beta = np.random.choice([-1.0,1.0])*resolution
    gamma = 0.
    # to rotate the bases relative to base coor, until the Denali opening does not contain the baseplate
    while (DenaliOpening.contains(Base)==True):
        gamma = gamma + Delta_Beta
        Base = rotate(Base, gamma, (Base.centroid.coords[0][0], Base.centroid.coords[0][1]) )
    print(gamma)
    return gamma



###############################################################################
##################### calculation parameters ##################################
###############################################################################
start = time.time()
print('Monte Carlo simulation for symmetrical base-plate and Tubecore')

# for Denali.HDA Solder Mask opening dimension
# x_Denali, dx_Denali, y_Denali, dy_Denali = 2.25, 0.1, 4.785, 0.1 # Denali.HDA bonding position and tol
x_Denali, dx_Denali, y_Denali, dy_Denali = 2.25, 0.1, 4.538, 0.1
x_DenaliOpening, y_DenaliOpening = x_Denali, y_Denali - 1.0 # design value of Denali.HDA Opening coor
beta_Denali, dbeta_Denali = 0.0, 1.0   # Denali bonding rotation 
dx_DenaliOpening, dy_DenaliOpening = 0.05, 0.05 # Denali.HDA opening position tol wrt Denali.HDA center
w_DenaliOpening, dw_DenaliOpening, l_DenaliOpening, dl_DenaliOpening = 3.3, 0.03, 3.3, 0.03 # unit Denali.HDA opening dimension & tol

 
# for copper baseplate (position of the base plate follows core and rod)
w_Base, dw_Base, l_Base, dl_Base = 3.06, 0.5, 3.06, 0.5 # unit baseplate dimension & tol specs
beta_Base, dbeta_Base = 0.0, 5.0 # rotation of base wrt to core.

# for rod 
Dy_Rod = 0.28   # offset of rod wrt to base plate (up mean positive)
x_Rod, dx_Rod, y_Rod, dy_Rod = x_DenaliOpening, 0.05, y_DenaliOpening + Dy_Rod, 0.05 # position of rod and fabrication tol (rod position wrt baseplate)
Dia_Rod, dDia_Rod = 1.62, 0.025 # rod diameter & tol


# for Core (inductor) dimension 
x_Core, dx_Core, y_Core, dy_Core = x_Rod, 0.1, y_Rod, 0.1  # core position follows rod's
w_Core, dw_Core, l_Core, dl_Core = 3.8 + 0.0, 1., 5.0 + 0.0, 1.  # tubecore width/length & tolerances specs
Core_Hole_Diameter, dCore_Hole_Diameter =  1.8, 0.1    # Tubecore hole diameter and tolerance
dx_Core_Hole, dy_Core_Hole = 0.1, 0.1  # Core hole position accuracy (fabrication) wrt Core center
chamfer = 0.3    # chamfer of the core
Con_Core2Rod, dCon_Core2Rod = 0., 0.1 # concentricity of the core hole wrt rod

# for clip landing lead
wClip, dwClip, lClip, dlClip = 2.44, 0.05, 0.5, 0.05
x_Clip, dx_Clip, y_Clip, dy_Clip = 2.25, 0.1, 0.715, 0.1  # positioning and accuracies
beta_Clip, dbeta_Clip = 0, 1.0 # allow rotation of clip 


#package dimension
x_Pkg, y_Pkg = 5.0, 4.5
w_Pkg, l_Pkg = 10.0, 9.0
saw_tol = 0.05


# decomment to check ideal component gemetries
# to visualize the ideal component geometries
DenaliOpening_ideal = Rectangle(x_DenaliOpening, y_DenaliOpening, w_DenaliOpening, 
                                l_DenaliOpening, beta_Denali, x_Denali, y_Denali)
Base_ideal = Rectangle(x_DenaliOpening, y_DenaliOpening, w_Base, l_Base, beta_Base, x_DenaliOpening, y_DenaliOpening)
Rod_ideal = Point( (x_Rod, y_Rod) ).buffer(0.5*Dia_Rod)

Core_ideal = Tubecore(x_Rod, y_Rod, w_Core, l_Core, chamfer, 0., x_DenaliOpening, y_DenaliOpening)

Core_Hole_ideal = Point( (x_Rod, y_Rod) ).buffer(0.5*Core_Hole_Diameter)

Package_ideal = Rectangle(x_Pkg, y_Pkg, w_Pkg, l_Pkg, 0, x_Pkg, y_Pkg)
ClipPad_ideal = Rectangle(x_Clip, y_Clip, wClip, lClip, 0., x_Clip, y_Clip)


print('Generating ideal geometries')
plt.figure(0)
plt.title('Ideal geometries')
plt.plot(*DenaliOpening_ideal.exterior.xy, 'b-', label = 'Denali.HDA opening')
plt.plot(*Base_ideal.exterior.xy, 'r-', label = 'Base plate')
plt.plot(*Rod_ideal.exterior.xy, 'r-')
plt.plot(*Package_ideal.exterior.xy, 'g-', label = 'Package')
plt.plot(*ClipPad_ideal.exterior.xy, 'g-')
plt.plot(*Core_ideal.exterior.xy, 'y-', label = "Tubecore")
plt.plot(*Core_Hole_ideal.exterior.xy, 'y-')
plt.legend()
plt.show()


###############################################################################
################## Monte Carlo Simulation #####################################
################## Random Numbers Generation ##################################
###############################################################################
# for Monte carlo simulation sample size
N = int(1e6)
Nsigma = 3 

print('Generating random numbers, sample size = ', N, ' , number of sigma = ', Nsigma)
print('1. Tubecores & their holes')
# bonding of core decide the position 
Core_coor_x, Core_coor_y = Normal(x_Core, dx_Core, Nsigma, N), Normal(y_Core, dy_Core, Nsigma, N) 

# coordinate of core's hole follows core and hole position deviation, 
CoreHole_coor_x = Core_coor_x + Normal(0., dx_Core_Hole, Nsigma, N) 
CoreHole_coor_y = Core_coor_y + Normal(0., dy_Core_Hole, Nsigma, N) 

CoreHole_Diameter = Normal(Core_Hole_Diameter, dCore_Hole_Diameter, Nsigma,N)

Core_chamfer = Normal(chamfer,0, Nsigma, N)
Core_width = Normal(w_Core, dw_Core, Nsigma, N)
Core_length = Normal(l_Core, dl_Core, Nsigma, N)


print('2. Rod ')
Rod_Diameter = Normal(Dia_Rod, dDia_Rod, Nsigma, N)

'''
# to calculate free-play between rods and holes
Hole2Rod_FreePlay = np.vectorize(RadiusRand)
FreePlay_x = Hole2Rod_FreePlay( 0.5*(CoreHole_Diameter - Rod_Diameter) )[0]
FreePlay_y = Hole2Rod_FreePlay( 0.5*(CoreHole_Diameter - Rod_Diameter) )[1] '''

# rod position and variation due to concentricity between core-hole to rod
Rod_coor_x = CoreHole_coor_x + Normal(Con_Core2Rod, dCon_Core2Rod, Nsigma, N)           # FreePlay_x   # on top of core position, free play affect the rod position
Rod_coor_y = CoreHole_coor_y + Normal(Con_Core2Rod, dCon_Core2Rod, Nsigma, N)           #FreePlay_y


print('3. Base plate')
Base_coor_x = Rod_coor_x + Normal(0, dx_Rod, Nsigma, N) # base plate follows rod. Need to compensate the rod offset
Base_coor_y = Rod_coor_y + Normal(-Dy_Rod, dy_Rod, Nsigma, N) 
Base_beta = Normal(beta_Base, dbeta_Base, Nsigma, N)  # rotation of base wrt core 

Base_width = Normal(w_Base, dw_Base, Nsigma, N)
Base_length = Normal(l_Base, dl_Base, Nsigma, N)


print('4. Denali & its opening')
Denali_coor_x = Normal(x_Denali, dx_Denali, Nsigma, N)
Denali_coor_y = Normal(y_Denali, dy_Denali, Nsigma, N)
Denali_beta = Normal(beta_Denali, dbeta_Denali, Nsigma, N)

DenaliOpening_coor_x = Normal(x_DenaliOpening, dx_DenaliOpening, Nsigma, N)
DenaliOpening_coor_y = Normal(y_DenaliOpening, dy_DenaliOpening, Nsigma, N)

DenaliOpening_width = Normal(w_DenaliOpening, dw_DenaliOpening, Nsigma, N)
DenaliOpening_length = Normal(l_DenaliOpening, dl_DenaliOpening, Nsigma, N)

print('5. Package edges')
Package_coor_x, Package_coor_y = Normal(x_Pkg, 0, Nsigma, N), Normal(y_Pkg, 0, Nsigma, N)
Package_width, Package_length = Normal(w_Pkg, saw_tol, Nsigma, N), Normal(l_Pkg, saw_tol, Nsigma, N)

print('6. Clip lead')
Clip_coor_x, Clip_coor_y = Normal(x_Clip, dx_Clip, Nsigma, N), Normal(y_Clip, dy_Clip, Nsigma, N)
Clip_width, Clip_length = Normal(wClip, dwClip, Nsigma, N), Normal(lClip, dlClip, Nsigma, N)
Clip_rotation = Normal(beta_Clip, dbeta_Clip, Nsigma, N)

# To construct geometries of components
print('Defining components geometries')
# base plate
print('1. Base plates, including core fabrication rotation')
Base_F = np.vectorize(Rectangle)
BasePlate = Base_F(Base_coor_x, Base_coor_y, Base_width, Base_length, Base_beta, Rod_coor_x, Rod_coor_y)

# Denali.HDA opening
print('2. Denali.HDA opening, including bonding rotiaton of Denali')
DenaliOpening_F = np.vectorize(Rectangle)
DenaliOpening = DenaliOpening_F(DenaliOpening_coor_x, DenaliOpening_coor_y,
                                DenaliOpening_width, DenaliOpening_length,
                                Denali_beta, Denali_coor_x, Denali_coor_y)

# calculating base rotation on Denali Opening
print('To calculate the rotational angle of base on the Denali Opening')
gamma_F = np.vectorize(BaseRotation)
gamma = gamma_F(BasePlate, DenaliOpening, 0.05)
print('Rotational angle of the baseplate mean & stdev: ', gamma.mean(), ', ', gamma.std() )

# Tubecore
print('3. Defining tubecore geometry')
Tubecore_F = np.vectorize(Tubecore)
Core = Tubecore_F(Core_coor_x, Core_coor_y, Core_width, Core_length, Core_chamfer, gamma, Base_coor_x, Base_coor_y )

# Package
print('Defining package edge geometry')
Package_F = np.vectorize(Rectangle)
PackageEdge = Package_F(Package_coor_x, Package_coor_y, 
                        Package_width, Package_length, 
                        0., Package_coor_x, Package_coor_y)

# Clip landing 
print('Defining clip lead')
Clip_lead_F = np.vectorize(Rectangle)
Clip_lead = Clip_lead_F(Clip_coor_x, Clip_coor_y, Clip_width, Clip_length, Clip_rotation, Rod_coor_x, Rod_coor_y) # rotation of clip around rod

# To calculate conflict
print('Calculating conflict between tubecore and package edge')
conflict_F = np.vectorize(Conflict)
conflict_rate = np.sum(conflict_F(Core,PackageEdge))

print('Calclulating conflict between tubecore and clip')
conflict_Core_Clip = np.sum(conflict_F(Core, Clip_lead))

end = time.time()

print('Calculation complete')
print('**********************************')
print('Sample size: ', N)
print('Tubecore size: ', Core_width.mean(), ', ', Core_length.mean() )
print('Tubecore hole diameter: ', CoreHole_Diameter.mean() )
print('Rod diameter: ', Rod_Diameter.mean() )
print('Core hole to rod concentricity: ', dCon_Core2Rod)
print('Baseplate rotation wrt core: ', dbeta_Base)
print('conflict rate between core and package edge:', conflict_rate)
print('conflict rate between core and clip: ', conflict_Core_Clip)
print('Tubecore rotation mean: ', gamma.mean())
print('Tubecore rotation standard deviation: ', gamma.std())
print('Time taken: ', end-start)



# Visualization
plt.figure(1)
for i in np.random.randint(1,N, 100):
    plt.plot(*DenaliOpening[i].exterior.xy, 'b-')
    plt.plot(*BasePlate[i].exterior.xy, 'r-')
    #plt.plot(*Rod)
    plt.plot(*Core[i].exterior.xy, 'g-')
    plt.plot(*PackageEdge[i].exterior.xy, 'm-' )
    plt.plot(*Clip_lead[i].exterior.xy, '--')
#plt.plot(*Package.exterior.xy)
plt.legend(loc='best')
plt.show()



df = pd.DataFrame( {
    'Core coor X': Core_coor_x, 'Core coor Y': Core_coor_y, 
    'Core width': Core_width,  'Core length': Core_length, 
    'Core hole coor X': CoreHole_coor_x, 'Core hole coor Y': CoreHole_coor_y, 
    'Core hole diameter': CoreHole_Diameter, 
    'Rod coor X': Rod_coor_x, 'Rod coor Y': Rod_coor_y,
    'Rod diameter': Rod_Diameter, 
    'Base-plate coor X': Base_coor_x, 'Base-plate coor Y': Base_coor_y,
    'Base-plate width': Base_width, 'Base-plate length': Base_length,
    'Package width': Package_width,
    'Package length': Package_length,
    'Core rotation': gamma,
    'Core-Package conflict': conflict_F(Core, PackageEdge)
    } )
    

df.to_csv('ComponentData_1.csv', mode='a', index=False, header=True) # for appending new data into ComponentData.csv file.
print('Writing to csv file done')









'''
# 1. to define Denali.HDA solder mask opening and position
Denali_width = Normal(wD, dwD, N) #width of Denali.HDAs
LD = Normal(lD, dlD, N) # Length of Denali.HDAs
X_Denali = Normal(x_D, dx_posD, N) # x-position of Denali.HD on laminate including bonding position offset
Y_Denali = Normal(y_D, dy_posD, N) # y-position of Denali.HD on laminate

# 2. to define baseplate dimension and position on Denali.HD
WB = Normal(wB, dwB,N) #width of baseplates
LB = Normal(lB, dlB, N)  # length of baseplates
Rod_Dia = Normal(rod_dia, drod_dia, N) 

X_base = Normal(x_B, dx_posB, N)  # x-position of baseplates
Y_base = Normal(y_B, dy_posB, N)   # y-position of baseplates

# 2.1 rotation of baseplate on Denali.HDA top opening
print('Calculating rotation of base-plate')
Base_Beta = RAND(N)*beta(WD, WB, LD, LB, X_base, Y_base)
print('Mean baseplate rotation: ', Base_Beta.mean(), ' stdev:', Base_Beta.std() )

print('Constructing base-plate')
REC = np.vectorize(Rectangle)
Base_Plate = REC(X_base, Y_base, WB, LB, Base_Beta, X_base, Y_base) # to construct rectangles representing baseplate

# 3. to define Tubecore dimension when coupling to the baseplate
WC = Normal(wC, dwC, N)
LC = Normal(lC, dlC, N)
C_Dia = Normal(C_dia, dC_dia, N)
print('Calculating Tubecore rotation around copper rod')
alpha = RAND(N)*Normal(0, dTheta, N)  # rotation of tubecore relative to baseplate upon assembly. Unit = degrees
print('Mean rotation of tubecore around rod ', alpha.mean(), ', sigma: ', alpha.std() )

# 3.1 when tubecore with baseplate is placed on Denali.HDA. 
# positioning of rod - follow baseplate positioning
X_rod = X_base + Normal(0,dx_Br,N) 
Y_rod = Y_base + Normal(Diff_y_Br, dy_Br, N)
# position of core around rod position with placement allowance defined by core 0.5*(hole dia-rod dia)
X_core = X_rod + RAND(N)*0.5*(C_Dia - Rod_Dia) # positioning dist. of core w/ offset allowable due to hole-rod clearance
Y_core = Y_rod + RAND(N)*0.5*(C_Dia - Rod_Dia) #  positioning dist. of core w/ offset allowable.

# 3.2 Creating tubecore polygon with both rotations (Tubecore and baseplate together)
# 3.2.1 with Tubecore rotation relative to baseplate
print('Calculating Tubecore rotation relative to base-plate')
Tubecore_F = np.vectorize(Tubecore)
#Tubecore_RF = np.vectorize(rotate)
TUBECORE = Tubecore_F(X_core, Y_core, WC, LC, fillet, alpha, X_core, Y_core) # polygon with assembly rotation to the rod. Core rotation is around the rod only.

# 3.2.2 to include rotation due to baseplate rotation which rotate around X_Base, Y_Base (which is the rotational center of the baseplate)
TUBECORE_Gamma = np.array([])
for i in np.arange(N):
    Gamma = rotate(TUBECORE[i], Base_Beta[i], (X_base[i], Y_base[i]) )
    TUBECORE_Gamma = np.append(TUBECORE_Gamma, Gamma) 

# 4. to define the solder mask opening of U-clip
WUC = Normal(wUC, dwC, N)  # various width of U-clips
LUC = Normal(lUC, dlUC, N) # various length of U-clips
X_UC = Normal(x_UC, dx_posUC, N)  # positioning of U-Clip
Y_UC = Normal(y_UC, dy_posUC, N)

U_Clip = REC(X_UC, Y_UC, WUC, LUC, Normal(0,0,N), 0, 0 ) # no rotation

# 5. to define the solder mask opening of clip landing
WCLP = Normal(wCLP, dwCLP, N)  # various width of clip landing pad
LCLP = Normal(lCLP, dlCLP, N)
X_CLP = Normal(x_CLP, dx_CLP, N)
Y_CLP = Normal(y_CLP, dy_CLP, N)

CLP = REC(X_CLP, Y_CLP, WCLP, LCLP, Normal(0,0,N), 0,0) # no rotation

# 6. to define package boundary
Pkg_Edge = Rectangle(0.5*wPkg,0.5*lPkg, wPkg, lPkg, 0, 0, 0) # theoretical package edges
Pkg_Boundary = LineString([ (0,0), (0, lPkg) ])

print('Calculating conflicts')
# 7. to check conflict btw structures
CON_T2Pkg = np.array([])
CON_T2CLP = np.array([]) # empty array for conflict between tubecore and clip landing pad
CON_T2UClip = np.array([])
distance_TUBECORE2Pkg = np.array([])
distance_TUBECORE2CLP = np.array([])
    
# 7.1 to detect conflict between Tubecore and pakcage edge:
for j in np.arange(N):
    aa = Conflict(TUBECORE_Gamma[j], Pkg_Boundary)
    bb = Conflict(TUBECORE_Gamma[j], CLP[j])
    cc = Conflict(TUBECORE_Gamma[j], U_Clip[j])
    dd = TUBECORE_Gamma[j].distance(Pkg_Boundary)
    ee = TUBECORE_Gamma[j].distance(CLP[j])
    CON_T2Pkg = np.append(CON_T2Pkg, aa)
    CON_T2CLP = np.append(CON_T2CLP, bb)
    CON_T2U_Clip = np.append(CON_T2UClip, cc)
    distance_TUBECORE2Pkg = np.append(distance_TUBECORE2Pkg, dd)
    distance_TUBECORE2CLP = np.append(distance_TUBECORE2CLP, ee)
    
end = time.time()

print('Sample size: ', N)
print('Tubecore dimension: ', wC, ' * ', lC) 
print('Rate of Tubecore conflict with package edge: ', 1e6*np.sum(CON_T2Pkg)/N, 'ppm')
print('Rate of Tubecore conflict with U-clip bonding position: ', 1e6*np.sum(CON_T2UClip)/N, 'ppm' )
print('Rate of Tubecore conflict with clip landing pad: ', 1e6*np.sum(CON_T2CLP)/N, 'ppm')
print('Distance between Tubecore to Package boundary: mean = ', distance_TUBECORE2Pkg.mean(), ', min = ', distance_TUBECORE2Pkg.min() )
print('Distance between Tubecore to clip landing pad: mean = ', distance_TUBECORE2CLP.mean(), ', min = ', distance_TUBECORE2CLP.min() )
print('Time taken = ', end - start, 'secs')

###############################################################################
######################## Graphical Plotting ###################################
###############################################################################

plt.figure()
xp, yp = Pkg_Edge.exterior.xy 
plt.plot(xp, yp)
for i in np.random.randint(1,N, 100):
    x,y = TUBECORE_Gamma[i].exterior.xy
    xUC, yUC = U_Clip[i].exterior.xy
    xB, yB = Base_Plate[i].exterior.xy
    xCLP, yCLP = CLP[i].exterior.xy
    plt.plot(xB,yB)
    plt.plot(x,y)       
    plt.plot(xUC,yUC)
    plt.plot(xCLP, yCLP)
    
plt.show()
'''