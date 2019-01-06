from vpython import *
import numpy as np

N = 200
m, size = 4E-3/6E23, 31E-12*10# He atom are 10 times bigger for easiear collision but not too big for accuracy 
L = ((24.4E-3/(6E23))*N)**(1/3.0)/2 + size # 2L is the cubic container's original length, width, and height
k, T = 1.38E-23, 750.0# Boltzmann Constant and initial temperature
t, dt ,count = 0, 3E-13,0
vrms = (2*k*1.5*T/m)**0.5# the initial root mean square velocity 
stage = 0# stage number
atoms = []# list to store atoms
K, P, g = 0, 0, 9.8
v_W = L / (20000.0*dt)
momentum = 0
# histogram setting
deltav, dv = 50., 10.
Cv = 3/2*k
pressure = 0#累計壓力值，以利取平均
total_W = 0
oldpos = 0#紀錄bar2末端的位置
displace = 0#軸末端位移
check, stage = 0, 1

min_length, mid_length, max_length = 2*L, 3*L, 4*L

#initialization
PV_graph = graph(width = 600, align = 'right', title='PV graph')
Eint_graph = graph(width = 600, align = 'right', title='Eint graph')
W_graph = graph(width = 600, align = 'right', title='W graph')
PV = gcurve(color = color.green, graph = PV_graph)
Eint = gcurve(color = color.red, graph = Eint_graph)
W = gcurve(color = color.purple, graph = W_graph)
scene = canvas(width = 500, height = 500, background = vector(0.2,0.2,0), align = 'left')
container = box(length = 2*L, height = 2*L, width = 2*L, opacity = 0.2, color = color.yellow)
pistone = box(length = L/20, height = 2*L, width = 2*L, pos = vec(L,0,0), color = color.orange)#設定活塞
pistone_m = (2E-21)/g#活塞質量
bar1 = cylinder(pos = vec(L,L/3,L/3), radius = L/20, axis = vec(3*L,0,0))
bar2 = cylinder(pos = bar1.pos + bar1.axis, radius = L/20,axis = vec(3*L,0,0))
wheel = ring(pos = vec(6.25*L,L/3,L), radius=1.25*L, thickness = L/10, axis = vec(0,0,1), texture = textures.stones)
wheel_m = 2E-21
count = 0

old_v, new_v, total_Q = 0, 0, 0
p_a, v_a = np.zeros((N,3)), np.zeros((N,3)) # particle position array and particle velocity array, N particles and 3 for x, y, z 

def vcollision(a1p, a2p, a1v,a2v): # the function for handling velocity after collisions between two atoms 
    v1prime = a1v - (a1p - a2p) * sum((a1v-a2v)*(a1p-a2p)) / sum((a1p-a2p)**2)
    v2prime = a2v - (a2p - a1p) * sum((a2v-a1v)*(a2p-a1p)) / sum((a2p-a1p)**2)
    return v1prime, v2prime

for i in range(N):
    p_a[i] = [2 * L*random() - L, 2 * L*random() - L, 2 * L*random() - L] # particle is initially random positioned in container 
    
    if i == N-1: # the last atom is with yellow color and leaves a trail
        atom = sphere(pos=vector(p_a[i, 0], p_a[i, 1], p_a[i, 2]), radius = size, color = color.yellow, make_trail = True, retain = 50) 
    
    else: # other atoms are with random color and leaves no trail
        atom = sphere(pos=vector(p_a[i, 0], p_a[i, 1], p_a[i, 2]), radius = size, color = vector(random(), random(), random())) 
    ra = pi*random()
    rb = 2*pi*random()
    v_a[i] = [vrms*sin(ra)*cos(rb), vrms*sin(ra)*sin(rb), vrms*cos(ra)] # particle initially same speed but random direction 
    atoms.append(atom)

while (1):
# slotwidth for v histogram
    t += dt 
    rate(3000)
    p_a += v_a*dt # calculate new positions for all atoms
    
    for i in range(N): atoms[i].pos = vector(p_a[i, 0], p_a[i, 1], p_a[i, 2]) # to display atoms at new positions
    r_array = p_a - p_a[:,np.newaxis]# array for vector from one atom to another atom for all pairs of atoms
    rmag = np.sqrt(np.sum(np.square(r_array), -1)) # distance array between atoms for all pairs of atoms
    hit = np.less_equal(rmag,2*size) - np.identity(N) # if smaller than 2*size meaning these two atoms might hit each other 
    hitlist = np.sort(np.nonzero(hit.flat)[0]).tolist()# change hit to a list 
    
    for ij in hitlist:# i,j encoded as i*Natoms+j
        i, j = divmod(ij,N)# atom pair, i-th and j-th atoms, hit each other
        hitlist.remove(j*N+i)# remove j,i pair from list to avoid handling the collision twice
        
        if sum((p_a[i] - p_a[j])*(v_a[i] - v_a[j])) < 0 :# only handling collision if two atoms are approaching each other
            v_a[i], v_a[j] = vcollision(p_a[i], p_a[j], v_a[i], v_a[j]) # handle collision
#find collisions between the atoms and the walls, and handle their elastic collisions
    for i in range(N):
        
        if abs(p_a[i][0]) >= container.length/2 - size and p_a[i][0]*v_a[i][0] > 0 :
            v_a[i][0] = -v_a[i][0]
            
            if stage == 2 or stage == 1:
                v_a[i][0] -= 2*v_a[i][0]/abs(v_a[i][0])*4*v_W
            
            if stage == 4:
                v_a[i][0] += 2*v_a[i][0]/abs(v_a[i][0])*4*v_W
            momentum += abs(2*m*v_a[i][0])
        
        if abs(p_a[i][1]) >= container.height/2 - size and p_a[i][1]*v_a[i][1] > 0 :
            v_a[i][1] = -v_a[i][1]
            momentum += abs(2*m*v_a[i][1])
        
        if abs(p_a[i][2]) >= container.width/2 - size and p_a[i][2]*v_a[i][2] > 0 :
            v_a[i][2] = -v_a[i][2]
            momentum += abs(2*m*v_a[i][2])
    
    P = momentum/(dt*(8*L**2+4*2*L*container.length))#計算每個時間氣體壓力
    pressure += P
    bar1.pos.x = container.length/2#兩根轉軸帶動輪子
    bar2.pos = bar1.pos + bar1.axis
    bar1.axis = vec(4*L-bar1.pos.x, 3*L*sin(acos((4*L-bar1.pos.x)/4*L)), L/3)#讓軸末端保持在平面上
    bar2.axis = vec(4*L-bar1.pos.x, -3*L*sin(acos((4*L-bar1.pos.x)/4*L)), L/3)
    displace = bar2.pos.x + bar2.axis.x - oldpos
    oldpos = bar2.pos.x + bar2.axis.x
    wheel.rotate(angle = displace/wheel.radius, axis = vec(0,0,1), origin = vector(wheel.pos.x,wheel.pos.y - wheel.radius,wheel.pos.z))
    wheel.pos.x = bar2.pos.x + bar2.axis.x - wheel.radius
    
    if stage == 1:
        container.length = container.length + v_W*8*dt
        
        if container.length >= mid_length :
            stage = 2
        bar1.pos.x = container.length/2
        pistone.pos.x += v_W*4*dt
    
    if stage == 2:
        container.length = container.length+v_W*8*dt
        if container.length >= max_length:#膨脹到壓力小於一定量值後收縮
            stage = 3
        bar1.pos.x = container.length/2
        pistone.pos.x += v_W*4*dt
    
    if stage == 3:
        container.length = container.length-v_W*8*dt
        if container.length <= mid_length :
            stage = 4
        bar1.pos.x = container.length/2
        pistone.pos.x -= v_W*4*dt
    
    if stage == 4:
        container.length = container.length - v_W*8*dt
        bar1.pos.x = container.length/2
        pistone.pos.x -= v_W*4*dt
        if container.length <= min_length :
            print('efficiency:', total_W/total_Q)
            break
    
    for i in range(N):
        K += 1/2*m*v_a[i][0]**2
        K += 1/2*m*v_a[i][1]**2
        K += 1/2*m*v_a[i][2]**2
    T = K/(3*N*k/2)
    
    if stage == 1 and T <= 750:
        for i in range(N):
            for j in range(3):
                old_v = v_a[i][j]
                v_a[i][j] += 100*random()
                new_v = v_a[i][j]
                total_Q += 1/2*m*(new_v**2-old_v**2)
    V = 4*L**2*container.length
    Eint.plot(t, N*Cv*T)
    
    if count%50 == 0:
        if stage == 1 or stage == 2:
            w = pressure*v_W*dt*4*L**2
        elif stage == 3 or stage == 4:
            w = -pressure*v_W*dt*4*L**2
        PV.plot(V,pressure/50)
        W.plot(t,w)
        pressure = 0
        total_W += w
    K, P = 0, 0
    
    momentum = 0
    count += 1
        
