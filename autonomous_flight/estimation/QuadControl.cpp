#include "Common.h"
#include "QuadControl.h"
#include "Utility/SimpleConfig.h"
#include "Utility/StringUtils.h"
#include "Trajectory.h"
#include "BaseController.h"
#include "Math/Mat3x3F.h"

#ifdef __PX4_NUTTX
#include <systemlib/param/param.h>
#endif


void QuadControl::Init()
{
    BaseController::Init();

    integratedAltitudeError = 0;
    
    #ifndef __PX4_NUTTX

    ParamsHandle config = SimpleConfig::GetInstance();
   
    kpPosXY = config -> Get(_config + ".kpPosXY", 0);
    kpPosZ  = config -> Get(_config + ".kpPosZ",  0);
    KiPosZ  = config -> Get(_config + ".KiPosZ",  0);
     
    kpVelXY = config -> Get(_config + ".kpVelXY", 0);
    kpVelZ  = config -> Get(_config + ".kpVelZ",  0);

    kpBank = config -> Get(_config + ".kpBank", 0);
    kpYaw  = config -> Get(_config + ".kpYaw",  0);

    kpPQR = config -> Get(_config + ".kpPQR", V3F());

    maxDescentRate = config -> Get(_config + ".maxDescentRate", 100);
    maxAscentRate  = config -> Get(_config + ".maxAscentRate",  100);
    maxSpeedXY     = config -> Get(_config + ".maxSpeedXY",     100);
    maxAccelXY     = config -> Get(_config + ".maxHorizAccel",  100);

    maxTiltAngle = config -> Get(_config + ".maxTiltAngle", 100);

    minMotorThrust = config -> Get(_config + ".minMotorThrust",   0);
    maxMotorThrust = config -> Get(_config + ".maxMotorThrust", 100);

    #else
    param_get(param_find("MC_PITCH_P"), &Kp_bank);
    param_get(param_find("MC_YAW_P"),   &Kp_yaw);

    #endif
}


VehicleCommand QuadControl::GenerateMotorCommands(float collThrustCmd, V3F momentCmd)
{
    /*
    INPUTS:
        collThrustCmd: desired collective thrust [N]
        momentCmd:     desired rotation moment about each axis [N m]

    OUTPUT:
        set class member variable cmd (class variable for graphing) where cmd.desiredThrustsN[0..3]: motor commands, in [N]
    */

    float ratio = L / sqrt(2.f);

    float m_1 =  momentCmd.x / ratio;
    float m_2 =  momentCmd.y / ratio;
    float m_3 = -momentCmd.z / kappa;
    float m_4 =  collThrustCmd;

    float denom = 4.f;

    cmd.desiredThrustsN[0] = ( m_1 + m_2 + m_3 + m_4) / denom;
    cmd.desiredThrustsN[1] = (-m_1 + m_2 - m_3 + m_4) / denom;
    cmd.desiredThrustsN[2] = ( m_1 - m_2 - m_3 + m_4) / denom;
    cmd.desiredThrustsN[3] = (-m_1 - m_2 + m_3 + m_4) / denom;

    return cmd;
}


V3F QuadControl::BodyRateControl(V3F pqrCmd, V3F pqr)
{
    /*
    INPUTS:
        pqrCmd: desired body rates [rad/s]
        pqr:    current or estimated body rates [rad/s]

    OUTPUT:
        return a V3F containing the desired moments for each of the 3 axes
    */

    V3F momentCmd;
    V3F I;

    I.x = Ixx;
    I.y = Iyy;
    I.z = Izz;

    momentCmd = I * kpPQR * (pqrCmd - pqr);

    return momentCmd;
}


V3F QuadControl::RollPitchControl(V3F accelCmd, Quaternion<float> attitude, float collThrustCmd)
{
    /*
    INPUTS:
        accelCmd:      desired acceleration in global XY coordinates [m/s2]
        attitude:      current or estimated attitude of the vehicle
        collThrustCmd: desired collective thrust of the quad [N]

    OUTPUT:
        return a V3F containing the desired pitch and roll rates. The Z element of the V3F should be left at its default value (0)
    */

    V3F pqrCmd;

    Mat3x3F R = attitude.RotationMatrix_IwrtB();

    if (collThrustCmd > 0) {
        float acceleration = -collThrustCmd / mass;

        // Roll
        float r_command = CONSTRAIN(
            accelCmd.x / acceleration,
            -maxTiltAngle,
            maxTiltAngle
            );

        float r_error = r_command - R(0, 2);
        float r_k_p = kpBank * r_error;


        // Pitch
        float p_command = CONSTRAIN(
             accelCmd.y / acceleration,
            -maxTiltAngle,
             maxTiltAngle
        );

        float p_error = p_command - R(1, 2);
        float p_k_p = kpBank * p_error;

        // Command
        pqrCmd.x = ((R(1, 0) * r_k_p) - (R(0, 0) * p_k_p)) / R(2, 2);
        pqrCmd.y = ((R(1, 1) * r_k_p) - (R(0, 1) * p_k_p)) / R(2, 2);
        pqrCmd.z = 0;
    }

    else {
        pqrCmd.x = 0;
        pqrCmd.y = 0;
        pqrCmd.z = 0;
    }

    return pqrCmd;
}


float QuadControl::AltitudeControl(float posZCmd, float velZCmd, float posZ, float velZ, Quaternion<float> attitude, float accelZCmd, float dt)
{
    /*
    INPUTS:
        posZCmd, velZCmd: desired vertical position and velocity in NED [m]
        posZ, velZ:       current vertical position and velocity in NED [m]
        accelZCmd:        feed-forward vertical acceleration in NED [m/s2]
        dt:               the time step of the measurements [seconds]

    OUTPUT:
         a collective thrust command in [N]
    */


    Mat3x3F R = attitude.RotationMatrix_IwrtB();

    float thrust = 0;

    float z_error     = posZCmd - posZ;
    float z_dot_error = velZCmd - velZ;

    integratedAltitudeError += z_error * dt;

    float z_k_p = kpPosZ * z_error;
    float z_k_i = KiPosZ * integratedAltitudeError;
    float z_k_d = kpVelZ * z_dot_error + velZ;

    float u_bar = z_k_p + z_k_i + z_k_d + accelZCmd;
    float force = (u_bar - CONST_GRAVITY) / R(2, 2);

    thrust = -mass * CONSTRAIN(
        force,
        -maxAscentRate  / dt,
         maxDescentRate / dt
    );
  
    return thrust;
}


V3F QuadControl::LateralPositionControl(V3F posCmd, V3F velCmd, V3F pos, V3F vel, V3F accelCmdFF)
{
    /*
    INPUTS:
        posCmd:     desired position, in NED [m]
        velCmd:     desired velocity, in NED [m/s]
        pos:        current position, NED [m]
        vel:        current velocity, NED [m/s]
        accelCmdFF: feed-forward acceleration, NED [m/s2]

    OUTPUT:
        return a V3F with desired horizontal accelerations. The Z component should be 0
    */

    accelCmdFF.z = 0;
    velCmd.z     = 0;
    posCmd.z     = pos.z;

    V3F accelCmd = accelCmdFF;

    // Position
    V3F k_p_position;

    k_p_position.x = kpPosXY;
    k_p_position.y = kpPosXY;
    k_p_position.z = 0.f;

    // Velocity
    V3F k_p_velocity;

    k_p_velocity.x = kpVelXY;
    k_p_velocity.y = kpVelXY;
    k_p_velocity.z = 0.f;

    // Command
    V3F pos_command = posCmd.mag() > maxAccelXY ? velCmd.norm() * maxAccelXY : posCmd;
    V3F vel_command = velCmd.mag() > maxSpeedXY ? velCmd.norm() * maxSpeedXY : velCmd;

    accelCmd += (k_p_position * (pos_command - pos)) + (k_p_velocity * (vel_command - vel));

    return accelCmd;
}


float QuadControl::YawControl(float yawCmd, float yaw)
{
    /*
    INPUTS: 
        yawCmd: commanded yaw [rad]
        yaw:    current yaw [rad] 

    OUTPUT:
        return a desired yaw rate [rad/s]
    */

    float yawRateCmd = 0;
    float rad_angle  = yawCmd > 0 ? fmodf(yawCmd, 2 * F_PI) : -fmodf(-yawCmd, 2 * F_PI);
    float yaw_error  = rad_angle - yaw;
    
    if (yaw_error > F_PI) {
        yaw_error -= 2 * F_PI;
    }

    if (yaw_error < -F_PI) {
        yaw_error += 2 * F_PI;
    }

    yawRateCmd = kpYaw * yaw_error;

    return yawRateCmd;
}


VehicleCommand QuadControl::RunControl(float dt, float simTime)
{
    curTrajPoint = GetNextTrajectoryPoint(simTime);

    float collThrustCmd = AltitudeControl(curTrajPoint.position.z, curTrajPoint.velocity.z, estPos.z, estVel.z, estAtt, curTrajPoint.accel.z, dt);
    float thrustMargin  = .1f*(maxMotorThrust - minMotorThrust);

    collThrustCmd = CONSTRAIN(collThrustCmd, (minMotorThrust+ thrustMargin)*4.f, (maxMotorThrust-thrustMargin)*4.f);
  
    V3F desAcc   = LateralPositionControl(curTrajPoint.position, curTrajPoint.velocity, estPos, estVel, curTrajPoint.accel);
    V3F desOmega = RollPitchControl(desAcc, estAtt, collThrustCmd);

    desOmega.z = YawControl(curTrajPoint.attitude.Yaw(), estAtt.Yaw());

    V3F desMoment = BodyRateControl(desOmega, estOmega);

    return GenerateMotorCommands(collThrustCmd, desMoment);
}