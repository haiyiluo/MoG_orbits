'''input position and velocity for a test asteroid
    output: reference orbital elements at reference date'''
import numpy as np 

MU=1
def orbit_elements(lines: list):

    
    vx = (lines[3])*365.2568983/(2*np.pi)
    vy = (lines[4])*365.2568983/(2*np.pi)
    vz = (lines[5])*365.2568983/(2*np.pi)
    x= (lines[0])
    y= (lines[1])
    z= (lines[2])

    print("XYZ", x, y, z)
    #semi-major axis
    v=(vx**2+vy**2+vz**2)**0.5
    r=(x**2+y**2+z**2)**0.5
    a=1/(2/r-v**2/MU)

    #eccentricity
    r_mat=np.array([x,y,z])
    v_mat=np.array([vx, vy, vz])
    h=np.cross(r_mat, v_mat)
    
    h_abs=(h[0]**2+h[1]**2+h[2]**2)**0.5
    print('h_,agnitude', h_abs)
    e=(1-h_abs**2/(MU*a))**0.5
    # e_abs=(e[0]**2+e[1]**2+e[2]**2)**0.5

    #inclination
    i=np.arccos(h[2]/h_abs)

    #OMEGA
    k=[0,0,1]
    n=np.cross(k,h)
    n_abs=(n[0]**2+n[1]**2+n[2]**2)**0.5
    #Omega=sign((n[1]/n_abs), (n[0]/n_abs))
    if (n[0]/n_abs)>0:
        Omega=np.arcsin((n[1]/n_abs))%(2*np.pi)
    elif (n[0]/n_abs)<0:
        Omega=np.pi-np.arcsin((n[1]/n_abs))

    #omega
    u_cap=np.arcsin(z/(r*np.sin(i)))
    uy=r_mat[2]/(r*np.sin(i))
    ux=(r_mat[0]*np.cos(Omega)+r_mat[1]*np.sin(Omega))/r
    if ux >0:
        u_cap=np.arcsin(uy)%(2*np.pi)
    elif (ux)<0:
        u_cap=np.pi-np.arcsin(uy)
    vv=np.arccos(1/e*(a*(1-e**2)/r-1))
    vvx=1/e*(a*(1-e**2)/r-1)
    vvy=a*(1-e**2)/(h_abs*e)*np.dot(r_mat, v_mat)/r

    if vvx >0:
        vv=np.arcsin(vvy)%(2*np.pi)
    elif (vvx)<0:
        vv=np.pi-np.arcsin(vvy)
    omega=(u_cap-vv)%(2*np.pi)

    '''print(np.rad2deg(u_cap),'u_cap')
    print(np.rad2deg(vv),'vv')'''
    #omega=sign(np.sin(u_cap-vv)y, np.cos(u_cap-vv)x)
    
        
    #M
    e_cap=np.arccos((a*e+r*np.cos(vv))/a)
    Ex=(a*e+r*np.cos(vv))/a
    Ey=(r*np.sin(vv))/(a*(1-e**2)**0.5)
    
    if Ex >0:
        e_cap=np.arcsin(Ey)%(2*np.pi)
    elif (Ex)<0:
        e_cap=np.pi-np.arcsin(Ey)

    m_cap=e_cap-e*np.sin(e_cap)

    print ('a', a, 'e', e, 'i', np.rad2deg(i), 'Omega', np.rad2deg(Omega), 'omega', np.rad2deg(omega),'M',np.rad2deg(m_cap))
    return  a, e,  np.rad2deg(i), np.rad2deg(Omega), np.rad2deg(omega), np.rad2deg(m_cap)
