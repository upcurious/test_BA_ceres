#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <chrono>
#include <opencv2/core/eigen.hpp>
#include <eigen3/Eigen/Dense>
using namespace std;
using namespace Eigen;
static Eigen::Vector3d R2ypr(const Eigen::Matrix3d &R)
{
    Eigen::Vector3d n = R.col(0);
    Eigen::Vector3d o = R.col(1);
    Eigen::Vector3d a = R.col(2);

    Eigen::Vector3d ypr(3);
    double y = atan2(n(1), n(0));
    double p = atan2(-n(2), n(0) * cos(y) + n(1) * sin(y));
    double r = atan2(a(0) * sin(y) - a(1) * cos(y), -o(0) * sin(y) + o(1) * cos(y));
    ypr(0) = y;
    ypr(1) = p;
    ypr(2) = r;

    return ypr ;/// M_PI * 180.0;
}
template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 3, 3> skewSymmetric(const Eigen::MatrixBase<Derived> &q)
{
    Eigen::Matrix<typename Derived::Scalar, 3, 3> ans;
    ans << typename Derived::Scalar(0), -q(2), q(1),
	q(2), typename Derived::Scalar(0), -q(0),
	-q(1), q(0), typename Derived::Scalar(0);
    return ans;
}
class PoseLocalParameterization : public ceres::LocalParameterization
{
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
    virtual bool ComputeJacobian(const double *x, double *jacobian) const;
    virtual int GlobalSize() const { return 7; };
    virtual int LocalSize() const { return 6; };
};
template <typename Derived>
static Eigen::Quaternion<typename Derived::Scalar> deltaQ(const Eigen::MatrixBase<Derived> &theta)
{
    typedef typename Derived::Scalar Scalar_t;

    Eigen::Quaternion<Scalar_t> dq;
    Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
    half_theta /= static_cast<Scalar_t>(2.0);
    dq.w() = static_cast<Scalar_t>(1.0);
    dq.x() = half_theta.x();
    dq.y() = half_theta.y();
    dq.z() = half_theta.z();
    return dq;
}
bool PoseLocalParameterization::Plus(const double *x, const double *delta, double *x_plus_delta) const
{
  Eigen::Map<const Eigen::Vector3d> _p(x);
  Eigen::Map<const Eigen::Quaterniond> _q(x + 3);

  Eigen::Map<const Eigen::Vector3d> dp(delta);

  Eigen::Quaterniond dq = deltaQ(Eigen::Map<const Eigen::Vector3d>(delta + 3));

  Eigen::Map<Eigen::Vector3d> p(x_plus_delta);
  Eigen::Map<Eigen::Quaterniond> q(x_plus_delta + 3);

  p = _p + dp;
  q = (_q * dq).normalized();

  return true;
}
bool PoseLocalParameterization::ComputeJacobian(const double *x, double *jacobian) const
{
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> j(jacobian);
    j.topRows<6>().setIdentity();
    j.bottomRows<1>().setZero();

    return true;
}
class ProjectionFactor : public ceres::SizedCostFunction<2, 7, 3>
{
  public:
  ProjectionFactor(const Eigen::Vector2d &_uv_i, const double &_uv_i_depth,double _fx, double _fy, double _cx, double _cy);
  
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;
    void check(double **parameters);
    Eigen::Vector2d uv_i;
    double uv_i_depth;
    double fx,  fy,  cx,  cy;
    Eigen::Matrix2d sqrt_info;
};
ProjectionFactor::ProjectionFactor(const Eigen::Vector2d &_uv_i, const double &_uv_i_depth,double _fx, double _fy, double _cx, double _cy) : uv_i(_uv_i),
uv_i_depth(_uv_i_depth),fx(_fx),fy(_fy),cx(_cx),cy(_cy)
{
  sqrt_info = fx / 1.5 * Eigen::Matrix2d::Identity();
};

bool ProjectionFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    Eigen::Vector3d Pi(parameters[0][0], parameters[0][1], parameters[0][2]);
    Eigen::Quaterniond Qi(parameters[0][6],parameters[0][3],parameters[0][4], parameters[0][5]);
    Eigen::Vector3d Ptw(parameters[1][0], parameters[1][1], parameters[1][2]);
    Matrix3d Ri = Qi.toRotationMatrix();
    Eigen::Vector3d CPi = Ri*Ptw + Pi;
    
    Eigen::Map<Eigen::Vector2d> residual(residuals);
    residual << uv_i(0) - (fx*CPi(0)/CPi(2) + cx),
                uv_i(1) - (fy*CPi(1)/CPi(2) + cy);
    
    residual = sqrt_info*residual;
    if (jacobians)
    {
        Eigen::Matrix<double, 2, 3> reduce(2, 3);

        reduce << fx / CPi(2), 0, -fx*CPi(0) / (CPi(2) * CPi(2)),
            0, fy / CPi(2), -fy*CPi(1) / (CPi(2) * CPi(2));
	    
	reduce = sqrt_info*reduce;
	
        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);

	    
	    jacobian_pose_i.block<2,3>(0,0) = -reduce;
            jacobian_pose_i.block<2,3>(0,3) = reduce*Ri*skewSymmetric(Ptw);
	    jacobian_pose_i.rightCols<1>().setZero();
        }
        if (jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> jacobian_pose_j(jacobians[1]);

	    jacobian_pose_j = -reduce*Ri;
        }
    }
    return true;
}

class FeaturePerFrame
{
  public:
    FeaturePerFrame(Vector2d _uv,double _depth):uv(_uv),depth(_depth){}
    Vector2d uv;
    double depth;
};
class Observation
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    Observation ( int mpt_id,int _start_frame,Eigen::Vector3d _ptw)
        :mpt_id_ ( mpt_id ),start_frame(_start_frame),ptw(_ptw){}
    Eigen::Vector3d ptw;
    int mpt_id_;
    int start_frame;
    vector<pair<FeaturePerFrame,int>> cam_id_;
};
void createData ( int n_mappoints, int n_cameras, double fx, double fy, double cx, double cy,
                  double height, double width,
                  std::vector<Eigen::Vector3d>& mappoints, std::vector<Eigen::Isometry3d>& cameras,
                  std::vector<Observation>& observations )
{
    
    const double angle_range = 0.1;
    const double x_range = 1.0;
    const double y_range = 1.0;
    const double z_range = 0.5;

    
    const double x_min = -5.0;
    const double x_max = 5.0;
    const double y_min = -5.0;
    const double y_max = 5.0;
    const double z_min = 0.6;
    const double z_max = 8.0;

    
    cv::RNG rng ( cv::getTickCount() );

    
    Eigen::Matrix3d Rx, Ry, Rz;
    Eigen::Matrix3d R; 
    Eigen::Vector3d t;
    for ( int i = 0; i < n_cameras; i ++ ) {
        
        double tz = rng.uniform ( -angle_range, angle_range );
        double ty = rng.uniform ( -angle_range, angle_range );
        double tx = rng.uniform ( -angle_range, angle_range );

        Rz << cos ( tz ), -sin ( tz ), 0.0,
           sin ( tz ), cos ( tz ), 0.0,
           0.0, 0.0, 1.0;
        Ry << cos ( ty ), 0.0, sin ( ty ),
           0.0, 1.0, 0.0,
           -sin ( ty ), 0.0, cos ( ty );
        Rx << 1.0, 0.0, 0.0,
           0.0, cos ( tx ), -sin ( tx ),
           0.0, sin ( tx ), cos ( tx );
        R = Rz * Ry * Rx;

        
        double x = rng.uniform ( -x_range, x_range );
        double y = rng.uniform ( -y_range, y_range );
        double z = rng.uniform ( -z_range, z_range );
        t << x, y, z;

        
        Eigen::Isometry3d cam;
	cam.linear() = R;
	cam.translation() = t;
        cameras.push_back ( cam );
    } 


    
    std::vector<Eigen::Vector3d> tmp_mappoints;
    
    for ( int i = 0; i < n_mappoints; i ++ ) {
        double x = rng.uniform ( x_min, x_max );
        double y = rng.uniform ( y_min, y_max );
        double z = rng.uniform ( z_min, z_max );
        tmp_mappoints.push_back ( Eigen::Vector3d ( x,y,z ) );
    }

    
    for ( int i = 0; i < n_mappoints; i ++ ) {
        const Eigen::Vector3d& ptw = tmp_mappoints.at ( i );
        int n_obs = 0.0;
        for ( int nc = 0; nc < n_cameras; nc ++ ) {
            const Eigen::Isometry3d& cam_pose = cameras[nc];
            
            const Eigen::Vector3d ptc = cam_pose * ptw;
            Eigen::Vector2d uv (
                fx*ptc[0]/ptc[2] + cx,
                fy*ptc[1]/ptc[2] + cy
            );

            if ( uv[0]<0 || uv[1]<0 || uv[0]>=width || uv[1]>=height || ptc[2] < 0.1 ) {
                continue;
            }
            n_obs ++;
        }

        if ( n_obs < 2 ) {
            continue;
        }

        mappoints.push_back ( ptw );
    }


    
    for ( size_t i = 0; i < mappoints.size(); i ++ ) {
        const Eigen::Vector3d& ptw = mappoints.at ( i );
	Observation ob (i,0,ptw);
        for ( int nc = 0; nc < n_cameras; nc ++ ) {
            const Eigen::Isometry3d& cam_pose = cameras[nc];

            
            const Eigen::Vector3d ptc = cam_pose * ptw;
            Eigen::Vector2d uv (
                fx*ptc[0]/ptc[2] + cx,
                fy*ptc[1]/ptc[2] + cy
            );
	    
	    FeaturePerFrame obs(uv,ptc[2]);
	    ob.cam_id_.push_back(make_pair(obs,nc));
            
        }
        observations.push_back ( ob );
    }
    
    mappoints.shrink_to_fit();
    cameras.shrink_to_fit();
    observations.shrink_to_fit();
};

void addNoise ( std::vector< Eigen::Vector3d >& mappoints, std::vector< Eigen::Isometry3d >& cameras, std::vector< Observation >& observations, double mpt_noise, double cam_trans_noise, double cam_rot_noise, double ob_noise )
{
    cv::RNG rng ( cv::getTickCount() );

    
    for ( size_t i = 0; i < mappoints.size(); i ++ ) {
        double nx = rng.gaussian ( mpt_noise );
        double ny = rng.gaussian ( mpt_noise );
        double nz = rng.gaussian ( mpt_noise );
        mappoints.at ( i ) += Eigen::Vector3d ( nx, ny, nz );
    }

    
	Eigen::Matrix3d Rx, Ry, Rz;
	Eigen::Matrix3d R; 
	Eigen::Vector3d t;
	for(size_t i = 0; i < cameras.size(); i ++)
	{
		
		if(i == 0)
			continue;
		
		double tz = rng.gaussian ( cam_rot_noise );
		double ty = rng.gaussian ( cam_rot_noise );
		double tx = rng.gaussian ( cam_rot_noise );
		
		Rz << cos ( tz ), -sin ( tz ), 0.0,
		sin ( tz ), cos ( tz ), 0.0,
		0.0, 0.0, 1.0;
		Ry << cos ( ty ), 0.0, sin ( ty ),
		0.0, 1.0, 0.0,
		-sin ( ty ), 0.0, cos ( ty );
		Rx << 1.0, 0.0, 0.0,
		0.0, cos ( tx ), -sin ( tx ),
		0.0, sin ( tx ), cos ( tx );
		R = Rz * Ry * Rx;
		
		
		double x = rng.gaussian ( cam_trans_noise );
		double y = rng.gaussian ( cam_trans_noise );
		double z = rng.gaussian ( cam_trans_noise );
		t << x, y, z;
		
		
		Eigen::Isometry3d cam_noise;
		cam_noise.linear() = R;
		cam_noise.translation() = t;
		cameras[i] = cameras[i]*cam_noise;
	}

	
	for(auto &map : observations)
	  for(auto &per_it : map.cam_id_)
	{
		double x = rng.gaussian ( ob_noise );
		double y = rng.gaussian ( ob_noise );
		per_it.first.uv += Eigen::Vector2d(x,y);
	}
}

double para_Pose[6][7];

int main ( int argc, char** argv )
{
    const int n_mappoints = 1000;
    const int n_cameras = 6;
	
    
    const double fx = 525.0;
    const double fy = 525.0;
    const double cx = 320.0;
    const double cy = 240.0;
    const double height = 640;
    const double width = 480;
	
    
    std::cout << "Start create data...\n";
    std::vector<Eigen::Vector3d> mappoints;
    std::vector<Eigen::Isometry3d> cameras;
    std::vector<Observation> observations;
    createData ( n_mappoints, n_cameras, fx, fy, cx, cy, height, width, mappoints, cameras, observations );
    std::cout << "Total mpt: " << mappoints.size() << "  cameras: " << cameras.size() << "  observations: " << observations.size() << std::endl;
    double mpt_noise = 0.5;
    double cam_trans_noise = 0.5;
    double cam_rot_noise = 0.1;
    double ob_noise = 2.0;
    
    
    std::vector<Eigen::Vector3d> noise_mappoints;
    noise_mappoints = mappoints;
    std::vector<Eigen::Isometry3d> noise_cameras;
    noise_cameras = cameras;
    std::vector<Observation> noise_observations;
    noise_observations = observations;
    addNoise(noise_mappoints, noise_cameras, noise_observations, mpt_noise, cam_trans_noise, cam_rot_noise, ob_noise );
    
    ceres::Problem problem;
    ceres::LossFunction *loss_function;
    
    loss_function = new ceres::CauchyLoss(1.0);
    double map_points[mappoints.size()][3];
    for(int i = 0;i<noise_cameras.size();i++)
    {
      Vector3d P = noise_cameras[i].translation();
      para_Pose[i][0] = P(0);
      para_Pose[i][1] = P(1);
      para_Pose[i][2] = P(2);
      
      Eigen::Quaterniond q{noise_cameras[i].linear()};
      para_Pose[i][3] = q.x();
      para_Pose[i][4] = q.y();
      para_Pose[i][5] = q.z();
      para_Pose[i][6] = q.w();
      
      ceres::LocalParameterization *local_parameterization = new PoseLocalParameterization();
      problem.AddParameterBlock(para_Pose[i], 7, local_parameterization);
      if(i == 0)
	problem.SetParameterBlockConstant(para_Pose[i]);
    }
    
    for ( auto &it_per_id :noise_observations)
    {
      
      Eigen::Vector3d ptw = it_per_id.ptw;
      map_points[it_per_id.mpt_id_][0] = ptw(0);
      map_points[it_per_id.mpt_id_][1] = ptw(1);
      map_points[it_per_id.mpt_id_][2] = ptw(2);
      problem.AddParameterBlock(map_points[it_per_id.mpt_id_], 3);
      for (auto &it_per_frame : it_per_id.cam_id_)
      {
	
	Eigen::Vector2d uv_j = it_per_frame.first.uv;
	double uv_j_depth = it_per_frame.first.depth;
	
	ProjectionFactor *f = new ProjectionFactor(uv_j,uv_j_depth,fx,fy,cx,cy);
	problem.AddResidualBlock(f, loss_function, para_Pose[it_per_frame.second],map_points[it_per_id.mpt_id_]);
      }
      
      
      
    }
    ceres::Solver::Options options;

    options.linear_solver_type = ceres::DENSE_SCHUR;
    //options.num_threads = 2;
    options.minimizer_progress_to_stdout = true;
    options.trust_region_strategy_type = ceres::DOGLEG;
    //options.max_num_iterations = NUM_ITERATIONS;
    //options.use_explicit_schur_complement = true;
    //options.minimizer_progress_to_stdout = true;
    //options.use_nonmonotonic_steps = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (summary.termination_type == ceres::CONVERGENCE || summary.final_cost < 5e-03)
    {
	    //cout << "vision only BA converge" << endl;
    }
    else
    {
	    cout << "vision only BA not converge " << endl;
	    return false;
    }
    cout<<summary.BriefReport() <<endl;
    
    std::vector<Eigen::Isometry3d> cam_Tw;
    
    for (int i = 0; i <= noise_cameras.size(); i++)
    {
      
	Eigen::Isometry3d cam_Tw_temp;
	
	cam_Tw_temp.linear() = Quaterniond(para_Pose[i][6], para_Pose[i][3], para_Pose[i][4], para_Pose[i][5]).normalized().toRotationMatrix();
	cam_Tw_temp.translation() = Vector3d(para_Pose[i][0],
				para_Pose[i][1],
				para_Pose[i][2]);
	cam_Tw.push_back(cam_Tw_temp);
    }
    double sum_rot_error = 0.0;
    double sum_trans_error = 0.0;
    for(size_t i = 0; i < cameras.size(); i ++)
    {
	    Eigen::Isometry3d opt_pose = noise_cameras[i];
	    Eigen::Isometry3d org_pose = cameras.at(i);
	    Eigen::Isometry3d pose_err = opt_pose * org_pose.inverse();
	    sum_rot_error += R2ypr(pose_err.linear()).norm();
	    sum_trans_error += pose_err.translation().norm();
    }
    std::cout << "pre Mean rot error: " << sum_rot_error / (double)(cameras.size())
    << "\tpre Mean trans error: " <<  sum_trans_error / (double)(cameras.size()) << std::endl;
    
     sum_rot_error = 0.0;
     sum_trans_error = 0.0;
    for(size_t i = 0; i < cameras.size(); i ++)
    {
	    Eigen::Isometry3d opt_pose = cam_Tw[i];
	    Eigen::Isometry3d org_pose = cameras.at(i);
	    Eigen::Isometry3d pose_err = opt_pose * org_pose.inverse();
	    sum_rot_error += R2ypr(pose_err.linear()).norm();
	    sum_trans_error += pose_err.translation().norm();
    }
    std::cout << "post Mean rot error: " << sum_rot_error / (double)(cameras.size())
    << "\tpost Mean trans error: " <<  sum_trans_error / (double)(cameras.size()) << std::endl;
    return 0;
}
