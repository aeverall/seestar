""" COORDINATE TRANSFORMATIONS 
This module contains methods that carry out coordinate transformations for left-handed coordinate systems.
    
Author:
    Payel Das
    
To do:
    Nothing I can think of at present.
"""

import numpy as np

# Converts proper motions in mas/year times distance in kpc to velocity in km/s [float]
pm2vel = 4.74057170372 

# Equatorial coordinate system constant [float]
ragp   = 3.36603292

# Equatorial coordinate system constant [float]
decgp  = 0.473477282

# Equatorial coordinate system constant [float]
lcp    = 2.145566725

def CartesianToSpherical(Cartesian):
    """ CARTESIAN (GALACTOCENTRIC) TO SPHERICAL (GALACTOCENTRIC)

    Arguments:
        Cartesian - Cartesian Galactocentric coordinates [vector]
            x      - (kpc, [-infty,infty])
            y      - (kpc, [-infty,infty])
            z      - (kpc, [-infty,infty])
            vx     - (km/s, [-infty,infty])
            vy     - (km/s, [-infty,infty])
            vz     - (km/s, [-infty,infty])
            
    Returns:
        Spherical - Spherical Galactocentric coordinates [vector]:
            r      - spherical radius (kpc, [0,infty])
            theta  - azimuth angle (rad,[-pi/2,pi/2])
            phi    - polar angle (rad,[0,2pi])
            vr     - radial velocity (km/s, [-infty,infty])
            vtheta - azimuthal velocity (km/s,[-infty,infty])
            vphi   - polar velocity (km/s,[-infty,infty])
    """

    # x,y,z -> r,theta,phi
    x     = Cartesian[:,0]
    y     = Cartesian[:,1]
    z     = Cartesian[:,2]
    r     = np.sqrt(x*x + y*y + z*z)
    projR = np.sqrt(x*x + y*y)
    theta = np.arccos(z/r)
    phi   = np.arctan2(y,x)
    theta[theta<0.] +=2.*np.pi
    
    if (len(Cartesian[0,:])==3):
        Spherical = np.column_stack((r,theta,phi))
        return Spherical
    else:
        # vx,vy,vz -> vr,vtheta,vphi
        vx = Cartesian[:,3]
        vy = Cartesian[:,4]
        vz = Cartesian[:,5]
        vr = (x*vx + y*vy + z*vz)/r
        vt = (z*vr - r*vz)/projR
        vp = r*np.sin(theta)*(vy*x-y*vx)/(projR*projR)     
        Spherical = np.column_stack((r,theta,phi,vr,vt,vp))
    return Spherical

def SphericalToCartesian(Spherical):
    
    """ SPHERICAL (GALACTOCENTRIC) TO CARTESIAN (GALACTOCENTRIC)

    Arguments:
        Spherical - Spherical Galactocentric coordinates [vector]:
            r      - spherical radius (kpc, [0,infty])
            theta  - azimuth angle (rad,[-pi/2,pi/2])
            phi    - polar angle (rad,[0,2pi])
            vr     - radial velocity (km/s, [-infty,infty])
            vtheta - azimuthal velocity (km/s,[-infty,infty])
            vphi   - polar velocity (km/s,[-infty,infty])
        
    Returns:
        Cartesian - Cartesian Galactocentric coordinates [vector]
            x      - (kpc, [-infty,infty])
            y      - (kpc, [-infty,infty])
            z      - (kpc, [-infty,infty])
            vx     - (km/s, [-infty,infty])
            vy     - (km/s, [-infty,infty])
            vz     - (km/s, [-infty,infty])     
    """

    # r,theta,phi -> x,y,z
    r  = Spherical[:,0]
    st = np.sin(Spherical[:,1])
    sp = np.sin(Spherical[:,2])
    ct = np.cos(Spherical[:,1])
    cp = np.cos(Spherical[:,2])
    x  = r*st*cp
    y  = r*st*sp
    z  = r*ct

    if (len(Spherical[0,:])==3):
        Cartesian = np.column_stack((x,y,z))
        return Cartesian
    else:
        # vr,vtheta,vphi -> vx,vy,vz
        vr = Spherical[:,3]
        vt = Spherical[:,4]
        vp = Spherical[:,5]
        vx = vr*st*cp - vt*ct*cp - vp*sp
        vy = vr*st*sp + vt*ct*sp + vp*cp
        vz = vr*ct - vt*st
        Cartesian= np.column_stack((x,y,z,vx,vy,vz))
        return Cartesian

def CartesianToPolar(Cartesian):
    
    """ CARTESIAN (GALACTOCENTRIC) TO POLAR (GALACTOCENTRIC)

    Arguments:
        Cartesian - Cartesian Galactocentric coordinates [vector]
            x      - (kpc, [-infty,infty])
            y      - (kpc, [-infty,infty])
            z      - (kpc, [-infty,infty])
            vx     - (km/s, [-infty,infty])
            vy     - (km/s, [-infty,infty])
            vz     - (km/s, [-infty,infty])  
        
    Returns:
        Polar - Polar Galactocentric coordinates [vector]
            R      - polar radius (kpc, [0,infty])
            phi    - polar angle (rad, [0,2pi])
            z      - (kpc, [-infty,infty])
            vR     - polar radial velocity (km/s, [-infty,infty])
            vphi   - polar velocity  (km/s, [-infty,infty])
            vz     - (km/s, [-infty,infty])    
    """
    
    # x,y,z -> R,phi,z
    R     = np.sqrt(Cartesian[:,0]*Cartesian[:,0]+Cartesian[:,1]*Cartesian[:,1])
    phi   = np.arctan2(Cartesian[:,1],Cartesian[:,0])
    z     = Cartesian[:,2]
    phi[phi<0.] += 2.*np.pi
    if (len(Cartesian[0,:])==3):
        Polar = np.column_stack((R,phi,z))
    else:
        # vx,vy,vz -> vR,vphi,vz
        cp   = np.cos(phi)
        sp   = np.sin(phi)
        vR   = Cartesian[:,3]*cp+Cartesian[:,4]*sp
        vphi = Cartesian[:,4]*cp-Cartesian[:,3]*sp
        vz   = Cartesian[:,5]
        Polar = np.column_stack((R,phi,z,vR,vphi,vz))
		
    return Polar

def PolarToCartesian(Polar):
    
    """ POLAR (GALACTOCENTRIC) TO CARTESIAN (GALACTOCENTRIC)

    Arguments:
        Polar - Polar Galactocentric coordinates [vector]
            R      - polar radius (kpc, [0,infty])
            phi    - polar angle (rad, [0,2pi])
            z      - (kpc, [-infty,infty])
            vR     - polar radial velocity (km/s, [-infty,infty])
            vphi   - polar velocity  (km/s, [-infty,infty])
            vz     - (km/s, [-infty,infty])    

    Returns:
        Cartesian - Cartesian Galactocentric coordinates [vector]
            x      - (kpc, [-infty,infty])
            y      - (kpc, [-infty,infty])
            z      - (kpc, [-infty,infty])
            vx     - (km/s, [-infty,infty])
            vy     - (km/s, [-infty,infty])
            vz     - (km/s, [-infty,infty])  
    """
	 
    # R,phi,z -> x,y,z
    cp = np.cos(Polar[:,1])
    sp = np.sin(Polar[:,1])
    x  = Polar[:,0] * cp
    y  = Polar[:,0] * sp
    z  = Polar[:,2]

    if (len(Polar[0,:])==3):
        Cartesian = np.column_stack((x,y,z))
    else:
        # vR,vphi,vz -> vx,vy,vz
        vx = Polar[:,3]*cp-Polar[:,4]*sp
        vy = Polar[:,4]*cp+Polar[:,3]*sp
        vz = Polar[:,5]
        Cartesian = np.column_stack((x,y,z,vx,vy,vz))
        
    return Cartesian

def GalacticToCartesian(Galactic,SolarPosition):
    """ GALACTIC (HELIOCENTRIC) TO CARTESIAN (GALACTOCENTRIC) GIVEN SOLAR POSITION
    
    Arguments: 
        Galactic - Galactic heliocentric coordinates [vector]
            l       - Galactic longitude (rad, [0,2pi])
            b       - Galactic latitude (rad, [-pi/2,pi/2])
            s       - distance (kpc, [0,infty])
            vlos    - line-of-sight velocity (km/s, [-infty,infty])
            mulcosb - projected longitudinal proper motion (mas/yr, [-infty,infty])
            mub     - latitudinal proper motion (mas/yr, [-infty,infty])
        SolarPosition - Polar coordinates of Sun [vector]
            R    - polar radius of Sun (kpc)
            phi  - polar angle of Sun (rad)
            z    - (kpc)
            vR   - polar radial velocity (km/s)
            vphi - polar velocity  (km/s)
            vz   - (km/s)
    
    Returns:
        Cartesian - Cartesian Galactocentric coordinates [vector]
            x      - (kpc, [-infty,infty])
            y      - (kpc, [-infty,infty])
            z      - (kpc, [-infty,infty])
            vx     - (km/s, [-infty,infty])
            vy     - (km/s, [-infty,infty])
            vz     - (km/s, [-infty,infty])  
    """   
    
    # l,b,s->x,y,z
    cl = np.cos(Galactic[:,0])
    sl = np.sin(Galactic[:,0])
    cb = np.cos(Galactic[:,1])
    sb = np.sin(Galactic[:,1])
    x  = SolarPosition[0]-Galactic[:,2]*cb*cl
    y  = Galactic[:,2]*cb*sl
    z  = Galactic[:,2]*sb+SolarPosition[1]

    if(len(Galactic[0,:])==3):
        Cartesian = np.column_stack((x,y,z))
    else:
        # vlos,mu_lcos(b),mu_b -> vx,vy,vz
        vl   = pm2vel*Galactic[:,2]*Galactic[:,4]
        vb   = pm2vel*Galactic[:,2]*Galactic[:,5]
        tmp2 = cb*Galactic[:,3]-sb*vb
        vx   = cl*tmp2-sl*vl+SolarPosition[2]
        vy   = sl*tmp2+cl*vl+SolarPosition[3]
        vz   = sb*Galactic[:,3]+cb*vb+SolarPosition[4]
        Cartesian = np.column_stack((x,y,z,-vx,vy,vz))
        
    return Cartesian
    
def CartesianToGalactic(Cartesian,SolarPosition):
    """ CARTESIAN (GALACTOCENTRIC) TO GALACTIC (HELIOCENTRIC) GIVEN SOLAR POSITION
    
    Arguments:
        Cartesian - Cartesian Galactocentric coordinates [vector]
            x      - (kpc, [-infty,infty])
            y      - (kpc, [-infty,infty])
            z      - (kpc, [-infty,infty])
            vx     - (km/s, [-infty,infty])
            vy     - (km/s, [-infty,infty])
            vz     - (km/s, [-infty,infty])          
        SolarPosition - Polar coordinates of Sun [vector]
            R    - polar radius of Sun (kpc)
            phi  - polar angle of Sun (rad)
            z    - (kpc)
            vR   - polar radial velocity (km/s)
            vphi - polar velocity  (km/s)
            vz   - (km/s)
    
    Returns:
        Galactic - Galactic heliocentric coordinates [vector]
            l       - Galactic longitude (rad, [0,2pi])
            b       - Galactic latitude (rad, [-pi/2,pi/2])
            s       - distance (kpc, [0,infty])
            vlos    - line-of-sight velocity (km/s, [-infty,infty])
            mulcosb - projected longitudinal proper motion (mas/yr, [-infty,infty])
            mub     - latitudinal proper motion (mas/yr, [-infty,infty])
    """   
	 
    # x,y,z->l,b,s
    tmp1 = SolarPosition[0]-Cartesian[:,0]
    tmp2 = Cartesian[:,1]
    tmp3 = Cartesian[:,2]-SolarPosition[1]
    s    = np.sqrt(tmp1*tmp1+tmp2*tmp2+tmp3*tmp3)
    l    = np.arctan2(tmp2,tmp1)
    b    = np.arcsin(tmp3/s)
    l[l<0.] += 2.*np.pi;    

    if(len(Cartesian[0,:])==3):
        Galactic = np.column_stack((l,b,s))
    else:
     	 # vx,vy,vz -> vlos,mu_lcos(b),mu_b
         vx  = -Cartesian[:,3]-SolarPosition[2]
         vy  = Cartesian[:,4]-SolarPosition[3]
         vz  = Cartesian[:,5]-SolarPosition[4]
         cl  = np.cos(l)
         sl  = np.sin(l)
         cb = np.cos(b)
         sb = np.sin(b)
         vlos = vx*cl*cb+vy*sl*cb+vz*sb;
         mul = (-vx*sl+vy*cl)/(pm2vel*s)
         mub = (-vx*cl*sb-vy*sl*sb+vz*cb)/(pm2vel*s)
         Galactic = np.column_stack((l,b,s,vlos,mul,mub))
         
    return Galactic

def PolarToGalactic(Polar,SolarPosition):
    """ POLAR (GALACTOCENTRIC) TO GALACTIC (HELIOCENTRIC) GIVEN SOLAR POSITION
    
    Arguments:
        Polar - Polar Galactocentric coordinates [vector]
            R      - polar radius (kpc, [0,infty])
            phi    - polar angle (rad, [0,2pi])
            z      - (kpc, [-infty,infty])
            vR     - polar radial velocity (km/s, [-infty,infty])
            vphi   - polar velocity  (km/s, [-infty,infty])
            vz     - (km/s, [-infty,infty])    
        SolarPosition - Polar coordinates of Sun [vector]
            R    - polar radius of Sun (kpc)
            phi  - polar angle of Sun (rad)
            z    - (kpc)
            vR   - polar radial velocity (km/s)
            vphi - polar velocity  (km/s)
            vz   - (km/s)
    
    Returns:
        Galactic - Galactic heliocentric coordinates [vector]
            l       - Galactic longitude (rad, [0,2pi])
            b       - Galactic latitude (rad, [-pi/2,pi/2])
            s       - distance (kpc, [0,infty])
            vlos    - line-of-sight velocity (km/s, [-infty,infty])
            mulcosb - projected longitudinal proper motion (mas/yr, [-infty,infty])
            mub     - latitudinal proper motion (mas/yr, [-infty,infty])
    """   
    
    Cartesian = PolarToCartesian(Polar)
    Galactic  = CartesianToGalactic(Cartesian,SolarPosition)
 
    return Galactic

### GALACTIC (HELIOCENTRIC) TO POLAR (GALACTOCENTRIC)
def GalacticToPolar(Galactic,SolarPosition):
    
    """ GALACTIC (HELIOCENTRIC) TO POLAR (GALACTOCENTRIC) GIVEN SOLAR POSITION
    
    Arguments:
        Galactic - Galactic heliocentric coordinates [vector]
            l       - Galactic longitude (rad, [0,2pi])
            b       - Galactic latitude (rad, [-pi/2,pi/2])
            s       - distance (kpc, [0,infty])
            vlos    - line-of-sight velocity (km/s, [-infty,infty])
            mulcosb - projected longitudinal proper motion (mas/yr, [-infty,infty])
            mub     - latitudinal proper motion (mas/yr, [-infty,infty])
        SolarPosition - Polar coordinates of Sun [vector]
            R    - polar radius of Sun (kpc)
            phi  - polar angle of Sun (rad)
            z    - (kpc)
            vR   - polar radial velocity (km/s)
            vphi - polar velocity  (km/s)
            vz   - (km/s)
    
    Returns:
        Polar - Polar Galactocentric coordinates [vector]
            R      - polar radius (kpc, [0,infty])
            phi    - polar angle (rad, [0,2pi])
            z      - (kpc, [-infty,infty])
            vR     - polar radial velocity (km/s, [-infty,infty])
            vphi   - polar velocity  (km/s, [-infty,infty])
            vz     - (km/s, [-infty,infty])    
    """   

    Cartesian = GalacticToCartesian(Galactic,SolarPosition)
    Polar     = CartesianToPolar(Cartesian)

    return Polar
    
def GalacticToEquatorial(Galactic):
    """ GALACTIC (HELIOCENTRIC) TO EQUATORIAL (HELIOCENTRIC)
    
    Arguments:
        Galactic - Galactic heliocentric coordinates [vector]
            l       - Galactic longitude (rad, [0,2pi])
            b       - Galactic latitude (rad, [-pi/2,pi/2])
            s       - distance (kpc, [0,infty])
            vlos    - line-of-sight velocity (km/s, [-infty,infty])
            mulcosb - projected longitudinal proper motion (mas/yr, [-infty,infty])
            mub     - latitudinal proper motion (mas/yr, [-infty,infty])
    
    Returns:
        Equatorial - Equatorial Galactocentric coordinates [vector]
            ra         - right ascencion (rad, [0,2pi])
            dec        - declination (rad, [-pi/2,pi/2])
            s          - distance (kpc, [0,infty])
            vlos       - line-of-sight velocity (km/s, [-infty,infty])
            muracosdec - projected right ascencion proper motion  (mas/yr, [-infty,infty])
            mudec      - declination proper motion (mas/yr, [-infty,infty]) 
    """
    
    # l,b,s => ra, dec, s
    l      = Galactic[:,0]
    b      = Galactic[:,1]
    cb     = np.cos(b)
    sb     = np.sin(b)
    dec    = np.arcsin(np.cos(decgp)*cb*np.cos(l-lcp)+sb*np.sin(decgp))
    ra     = ragp+np.arctan2(cb*np.sin(lcp-l),sb*np.cos(decgp)-cb*np.sin(decgp)*np.cos(l-lcp))
    ra[ra>2.*np.pi] -= 2.*np.pi
    if (len(Galactic[0,:])==3):
        Equatorial = np.column_stack([ra,dec,Galactic[:,2]])
    else:
        # vlos, mulcos(b), mub => vlos, muracos(dec), mudec
        cd         = np.cos(dec)
        sd         = np.sin(dec)
        A11        = (np.sin(decgp)*cd-np.cos(decgp)*sd*np.cos(ra-ragp))/cb
        A12        = -np.cos(decgp)*np.sin(ra-ragp)/cb
        A21        = (np.cos(decgp)*cd+np.sin(decgp)*sd*np.cos(ra-ragp)+sb*np.cos(lcp-l)*A11)/np.sin(lcp-l)
        A22        = (np.sin(decgp)*np.sin(ra-ragp)+sb*np.cos(lcp-l)*A12)/np.sin(lcp-l)
        index      = np.where(np.fabs(np.cos(lcp-l))>np.fabs(np.sin(lcp-l)))
        A21[index] = (sd[index]*np.sin(ra[index]-ragp)-sb[index]*np.sin(lcp-l[index])*A11[index])/np.cos(lcp-l[index])
        A22[index] =-(np.cos(ra[index]-ragp)+sb[index]*np.sin(lcp-l[index])*A12[index])/np.cos(lcp-l[index])
        Prod       = A11*A22-A12*A21
        Equatorial = np.column_stack((ra,dec,Galactic[:,2],Galactic[:,3],
                                      (A11*Galactic[:,4]-A21*Galactic[:,5])/Prod,
                                      (A22*Galactic[:,5]-A12*Galactic[:,4])/Prod))
        
    return Equatorial
    
def EquatorialToGalactic(Equatorial):
    """ EQUATORIAL (HELIOCENTRIC) TO GALACTIC (HELIOCENTRIC)
    
    Arguments:
        Equatorial - Equatorial Galactocentric coordinates [vector]
            ra         - right ascencion (rad, [0,2pi])
            dec        - declination (rad, [-pi/2,pi/2])
            s          - distance (kpc, [0,infty])
            vlos       - line-of-sight velocity (km/s, [-infty,infty])
            muracosdec - projected right ascencion proper motion  (mas/yr, [-infty,infty])
            mudec      - declination proper motion (mas/yr, [-infty,infty])  
    
    Returns: 
        Galactic - Galactic heliocentric coordinates [vector]
            l       - Galactic longitude (rad, [0,2pi])
            b       - Galactic latitude (rad, [-pi/2,pi/2])
            s       - distance (kpc, [0,infty])
            vlos    - line-of-sight velocity (km/s, [-infty,infty])
            mulcosb - projected longitudinal proper motion (mas/yr, [-infty,infty])
            mub     - latitudinal proper motion (mas/yr, [-infty,infty])
    """
    
    # ra, dec, s => l,b,s
    ra     = Equatorial[:,0]
    dec    = Equatorial[:,1]
    s      = Equatorial[:,2]
    cd     = np.cos(dec)
    sd     = np.sin(dec)
    b      = np.arcsin(np.sin(decgp)*sd+np.cos(decgp)*cd*np.cos(ra-ragp))
    l      = lcp-np.arctan2(cd*np.sin(ra-ragp),np.cos(decgp)*sd-np.sin(decgp)*cd*np.cos(ra-ragp))
    l[l<0] += 2.*np.pi;    
    if(len(Equatorial[0,:])==3):
        Galactic  = np.column_stack((l,b,s))
    else:
        # vlos, muracos(dec), mudec => vlos, mulcosb, mub
        vlos       = Equatorial[:,3]
        muracosd   = Equatorial[:,4]
        mudec      = Equatorial[:,5]
        cb         = np.cos(b)
        sb         = np.sin(b)
        A11        = (np.sin(decgp)*cd-np.cos(decgp)*sd*np.cos(ra-ragp))/cb
        A12        = -np.cos(decgp)*np.sin(ra-ragp)/cb
        A21        = (np.cos(decgp)*cd+np.sin(decgp)*sd*np.cos(ra-ragp)+sb*np.cos(lcp-l)*A11)/np.sin(lcp-l)
        A22        = (np.sin(decgp)*np.sin(ra-ragp)+sb*np.cos(lcp-l)*A12)/np.sin(lcp-l)   
        index      = np.where(np.fabs(np.cos(lcp-l)) > np.fabs(np.sin(lcp-l)))
        A21[index] = (sd[index]*np.sin(ra[index]-ragp)-sb[index]*np.sin(lcp-l[index])*A11[index])/np.cos(lcp-l[index])
        A22[index] = -(np.cos(ra[index]-ragp)+sb[index]*np.sin(lcp-l[index])*A12[index])/np.cos(lcp-l[index])
   
        Galactic = np.column_stack((l,b,s,vlos,A21*mudec+A22*muracosd,A11*mudec+A12*muracosd))
        
    return Galactic
    
def CalcaUVW(Galactic,R0):
    """ U, V, W, VELOCITIES (HELIOCENTRIC) FROM GALACTIC (HELIOCENTRIC) GIVEN SOLAR RADIUS (SCHOENRICH 2012)
    
    Arguments:
        Galactic - Galactic heliocentric coordinates [vector]
            l       - Galactic longitude (rad, [0,2pi])
            b       - Galactic latitude (rad, [-pi/2,pi/2])
            s       - distance (kpc, [0,infty])
            vlos    - line-of-sight velocity (km/s, [-infty,infty])
            mulcosb - projected longitudinal proper motion (mas/yr, [-infty,infty])
            mub     - latitudinal proper motion (mas/yr, [-infty,infty])
        R0 - polar radius of Sun (kpc)
        
    Returns:
        alpha - Galactic angle (rad) [vector]
        U     - Cartesian heliocentric velocity towards Galactic centre (km/s) [vector]
        V     - Cartesian heliocentric velocity in direction of solar rotation (km/s) [vector]
        W     - Cartesian heliocentral velocity in upwards direction (km/s) [vector]
    """
    l       = Galactic[:,0]
    b       = Galactic[:,1]
    s       = Galactic[:,2]
    vlos    = Galactic[:,3]
    mulcosb = Galactic[:,4]
    mub     = Galactic[:,5]
    alpha   = np.arctan2((s*np.sin(l)*np.cos(b)),(R0-s*np.cos(l)*np.cos(b)))
    U       = pm2vel*(-np.sin(b)*np.cos(l)*s*mub - np.sin(l)*s*mulcosb) + np.cos(b)*np.cos(l)*vlos
    V       = pm2vel*(-np.sin(b)*np.sin(l)*s*mub + np.cos(l)*s*mulcosb) + np.cos(b)*np.sin(l)*vlos
    W       = pm2vel*np.cos(b)*s*mub + np.sin(b)*vlos
    
    return(alpha,U,V,W)