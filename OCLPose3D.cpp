#include <fstream>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>
#include "OCLPose3D.h"


using namespace cv;


namespace ocl {


/************************************************************************
 * Definition of PrivatePose
 * - does all of the geometrical work
 ************************************************************************/
class Pose3D::PrivatePose3D
{
public:
	PrivatePose3D(Pose3D* iface) : iface(iface) {}
	PrivatePose3D& operator=(const PrivatePose3D& rhs)
	{
		camera_transform = rhs.camera_transform;
		inv_camera_transform = rhs.inv_camera_transform;
		intrinsics_transform = rhs.intrinsics_transform;
		project_transform = rhs.project_transform;
		inv_project_transform = rhs.inv_project_transform;
		return *this;
	}

	Pose3D* iface;

	Eigen::Isometry3d camera_transform;
	Eigen::Isometry3d inv_camera_transform;
	Eigen::Isometry3d intrinsics_transform;
	Eigen::Projective3d project_transform;
	Eigen::Projective3d inv_project_transform;

	Eigen::Isometry3d intrinsicsTransform() const
	{
		Eigen::Isometry3d m; m.setIdentity();
		m(0,0) = iface->m_focal_x;
		m(0,2) = iface->m_image_center_x;
		m(1,1) = iface->m_focal_y;
		m(1,2) = iface->m_image_center_y;
		return m;
	}

	Eigen::Matrix3d eigenRotation() const
	{ return camera_transform.rotation(); }


	Eigen::Vector3d eigenTranslation() const
	{ return camera_transform.translation(); }


	Eigen::Isometry3d eigenRotationTransform() const
	{
		Eigen::Isometry3d r = Eigen::Isometry3d::Identity();
		r.rotate(eigenRotation());
		return r;
	}


	Eigen::Vector4d projectToImage(const Eigen::Vector4d& p) const
	{
		if (iface->m_has_camera_params == false) {
			printf("You need to set camera params first!");
		}

		Eigen::Vector4d r = project_transform * p;
		r(0) /= r(2);
		r(1) /= r(2);
				
		return r;
	}


	void unprojectFromImage(const Eigen::Vector4d& p, Eigen::Vector4d& output) const
	{
		if (iface->m_has_camera_params == false) {
			printf("You need to set camera params first!");
		}
		
		Eigen::Vector4d r (p(0)*p(2), p(1)*p(2), p(2), 1);
		output = inv_project_transform * r;		
	}


	void unprojectFromKinectImage(const Eigen::Vector4d& p, Eigen::Vector4d& output) const
	{
		if (iface->m_has_camera_params == false) {
			printf("You need to set camera params first!");
		}
		
		Eigen::Vector4d r (p(0)*p(2), p(1)*p(2), p(2), 1);
		output = inv_project_transform * r;
		
	}


	void applyTransformBefore(const Eigen::Isometry3d& transform)
	{
		camera_transform = camera_transform * transform;
		computeProjectiveTransform();
	}


	void applyTransformAfter(const Eigen::Isometry3d& transform)
	{
		camera_transform = transform * camera_transform;
		computeProjectiveTransform();
	}


	void computeProjectiveTransform()
	{
		// y points downward in the image, upward in real world.
		// same for z.
		Eigen::Isometry3d to_opencv = Eigen::Isometry3d::Identity();
		to_opencv(1,1) = to_opencv(2,2) = -1;

		Eigen::Projective3d projection = Eigen::Projective3d::Identity();
		
		Eigen::Isometry3d intrinsics = intrinsicsTransform();
		inv_camera_transform = camera_transform.inverse();

		project_transform = intrinsics * projection * to_opencv * camera_transform;
		inv_project_transform = project_transform.inverse();
	}
	
	// http://eigen.tuxfamily.org/dox/StructHavingEigenMembers.html
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

private:
	PrivatePose3D(const PrivatePose3D&);
};



Pose3D :: Pose3D() :
impl(new PrivatePose3D(this)),
	m_focal_x(1), m_focal_y(1),
	m_image_center_x(0), m_image_center_y(0),
	m_has_camera_params(false)
{
	impl->camera_transform.setIdentity();
	impl->project_transform = impl->camera_transform;
	impl->inv_camera_transform = impl->camera_transform;
	impl->inv_project_transform = impl->project_transform.inverse();
}


Pose3D :: ~Pose3D()
{
	delete impl;
	impl = 0;
}


Pose3D :: Pose3D(const Pose3D& rhs)
	: impl(new PrivatePose3D(this))
{
	*this = rhs;
}


Pose3D& Pose3D :: operator=(const Pose3D& rhs)
{
	m_focal_x = rhs.m_focal_x;
	m_focal_y = rhs.m_focal_y;
	m_image_center_x = rhs.m_image_center_x;
	m_image_center_y = rhs.m_image_center_y;
	m_has_camera_params = rhs.m_has_camera_params;
	*impl = *rhs.impl;
	return *this;
}


void Pose3D :: setCameraParametersFromOpencv(const cv::Mat1d& cv_matrix)
{
	double fx = cv_matrix(0,0);
	double fy = cv_matrix(1,1);
	double cx = cv_matrix(0,2);
	double cy = cv_matrix(1,2);
	setCameraParameters(fx,fy,cx,cy);
	impl->computeProjectiveTransform();
}


void Pose3D :: toLeftCamera(const cv::Mat1d& cv_matrix,
	const cv::Mat1d& R,
	const cv::Mat1d& T)
{
	double fx = cv_matrix(0,0);
	double fy = cv_matrix(1,1);
	double cx = cv_matrix(0,2);
	double cy = cv_matrix(1,2);
	setCameraParameters(fx,fy,cx,cy);

	cv::Mat1d to_gl_base(3,3); setIdentity(to_gl_base);
	to_gl_base(1,1) = -1;
	to_gl_base(2,2) = -1;

	cv::Mat1d new_R = to_gl_base.inv() * R * to_gl_base;
	cv::Mat1d new_T = to_gl_base * (T);

	applyTransformBefore(toVec3d(new_T), new_R);
}


void Pose3D :: toRightCamera(const cv::Mat1d& cv_matrix,
	const cv::Mat1d& R,
	const cv::Mat1d& T)
{
	double fx = cv_matrix(0,0);
	double fy = cv_matrix(1,1);
	double cx = cv_matrix(0,2);
	double cy = cv_matrix(1,2);
	setCameraParameters(fx,fy,cx,cy);

	// OpenCV coords has y down and z toward scene.
	// OpenGL classical 3d coords has y up and z backwards
	// This is the transform matrix.
	cv::Mat1d to_gl_base(3,3); setIdentity(to_gl_base);
	to_gl_base(1,1) = -1;
	to_gl_base(2,2) = -1;

	cv::Mat1d new_R = to_gl_base.inv() * R.inv() * to_gl_base;
	cv::Mat1d new_T = to_gl_base * (-T);

	applyTransformBefore(toVec3d(new_T), new_R);
}


void Pose3D :: setCameraParameters(double fx, double fy, double cx, double cy)
{
	m_focal_x = fx;
	m_focal_y = fy;
	m_image_center_x = cx;
	m_image_center_y = cy;
	m_has_camera_params = true;
	impl->computeProjectiveTransform();
}


void Pose3D :: resetCameraTransform()
{
	impl->camera_transform.setIdentity();
	impl->computeProjectiveTransform();
}


void Pose3D :: setCameraTransform(const cv::Mat1d& tvec, const cv::Mat1d& rvec)
{
	cv::Mat1d to_open_cv (4,4);
	setIdentity(to_open_cv);
	to_open_cv(1,1) = -1;
	to_open_cv(2,2) = -1;
	cv::Mat1d from_open_cv = to_open_cv.inv();

	CvMat c_rvec = rvec;
	cv::Mat1d rot(3,3); CvMat c_rot = rot;
	
	cv::Mat1d H = cv::Mat1d(4,4);
	setIdentity(H);
	cv::Mat1d H_rot = H(Rect(0,0,3,3));
	rot.copyTo(H_rot);
	H(0,3) = tvec(0,0);
	H(1,3) = tvec(1,0);
	H(2,3) = tvec(2,0);
	
	H = from_open_cv * H * to_open_cv;

	cv::Mat1f Hf(4,4);
	for_all_rc(Hf)
		Hf(r,c) = H(r,c);
	setCameraTransform(Hf);
}


void Pose3D :: setCameraTransform(const cv::Mat1f& H)
{
	for_all_rc(H)
		impl->camera_transform(r,c) = H(r,c);
	impl->computeProjectiveTransform();
}


void Pose3D :: setCameraTransform(const cv::Mat1d& H)
{
	for_all_rc(H)
		impl->camera_transform(r,c) = H(r,c);
	impl->computeProjectiveTransform();
}


cv::Point3f Pose3D :: cameraTransform(const cv::Point3f& p) const
{
	Eigen::Vector3d ep; 
	toEigen(p, ep);
	ep = impl->camera_transform * ep;
	return toVec3f(ep);
}


cv::Mat1f Pose3D :: cvInvCameraTransform() const
{
	cv::Mat1f m(4,4);
	Eigen::Matrix4d eigen_m = impl->inv_camera_transform.matrix();
	toOpencv(eigen_m, m);
	return m;
}


const cv::Vec3f Pose3D :: cvTranslation() const
{ 
	return toVec3f((Eigen::Vector3d)impl->camera_transform.translation()); 
}


const cv::Vec3f Pose3D :: cvEulerRotation() const
{
	cv::Vec3f angles;
	Eigen::Matrix3d rotM =  impl->camera_transform.rotation().matrix().transpose();
	double xy = sqrt(double(sqr(rotM(0,0)) + sqr(rotM(0,1))));
	if (xy > std::numeric_limits<double>::epsilon() * 8.0)
	{
		angles(0) = atan2(double(rotM(1,2)), double(rotM(2,2)));
		angles(1) = atan2(double(-rotM(0,2)), double(xy));
		angles(2) = atan2(double(rotM(0,1)), double(rotM(0,0)));
	}
	else
	{
		angles(0) = atan2(double(-rotM(2,1)), double(rotM(1,1)));
		angles(1) = atan2(double(-rotM(0,2)), double(xy));
		angles(2) = 0;
	}
	return angles;
	// Eigen::Vector3d coeffs = impl->camera_transform.rotation().eulerAngles(0, 1, 2);
	// return toVec3f(coeffs);
}


const cv::Mat1f Pose3D :: cvCameraTransform() const
{
	cv::Mat1f m(4,4);
	toOpencv(impl->camera_transform.matrix(), m);
	return m;
}


const cv::Mat1d Pose3D :: cvCameraTransformd() const
{
	cv::Mat1d m(4,4);
	toOpencv(impl->camera_transform.matrix(), m);
	return m;
}

void Pose3D::cvRotationMatrixTranslation(cv::Mat1d& translation, cv::Mat1d& rotation) const
{
	translation.create(3,1);
	rotation.create(3,3);
	cv::Mat1f H = cvCameraTransform();
	for(int r = 0; r < 3; ++r)
		for(int c = 0; c < 3; ++c)
			rotation(r,c) = H(r,c);
	translation(0,0) = H(0,3);
	translation(1,0) = H(1,3);
	translation(2,0) = H(2,3);
}


cv::Mat1f Pose3D :: cvProjectionMatrix() const
{
	cv::Mat1f m(4,4);
	toOpencv(impl->project_transform.matrix(), m);
	return m;
}


cv::Mat1f Pose3D :: cvInvProjectionMatrix() const
{
	cv::Mat1f m(4,4);
	toOpencv(impl->inv_project_transform.matrix(), m);
	return m;
}


cv::Point3f Pose3D :: invCameraTransform(const cv::Point3f& p) const
{
	Eigen::Vector3d ep; toEigen(p, ep);
	ep = impl->inv_camera_transform * ep;
	return toVec3f(ep);
}


cv::Vec4f Pose3D :: cvQuaternionRotation() const
{
	Eigen::Quaterniond q(impl->camera_transform.rotation().matrix());
	return Vec4f(q.x(), q.y(), q.z(), q.w());
}


float Pose3D :: determinant() const
{
	return impl->camera_transform.matrix().determinant();
}

void Pose3D :: invert()
{
	if(impl->camera_transform.matrix().determinant() > 1e-3) {
		printf("Matrix is not invertible!");
	}
	impl->camera_transform = impl->camera_transform.inverse();
	impl->computeProjectiveTransform();
}


Pose3D Pose3D :: inverted() const
{
	Pose3D p = *this;
	p.invert();
	return p;
}


cv::Point3f Pose3D :: projectToImage(const cv::Point3f& p) const
{
	Eigen::Vector4d ep; toEigen(p, ep);
	return toVec3f(impl->projectToImage(ep));
}


void Pose3D :: projectToImage(const cv::Mat3f& voxels, const cv::Mat1b& mask, cv::Mat3f& pixels) const
{
	Eigen::Vector4d epix;
	Eigen::Vector4d evox;
	evox(3) = 1; // w does not change.

	for (int r = 0; r < voxels.rows; ++r)
	{
		const Vec3f* voxels_data = voxels.ptr<Vec3f>(r);
		const uchar* mask_data = mask.ptr<uchar>(r);
		Vec3f* pixels_data = pixels.ptr<Vec3f>(r);
		for (int c = 0; c < voxels.cols; ++c)
		{
			if (!mask_data[c])
				continue;
			evox(0) = voxels_data[c][0];
			evox(1) = voxels_data[c][1];
			evox(2) = voxels_data[c][2];
			epix = impl->project_transform * evox;
			pixels_data[c][0] = epix(0)/epix(2);
			pixels_data[c][1] = epix(1)/epix(2);
			pixels_data[c][2] = epix(2);
		}
	}
}


void Pose3D :: unprojectFromImage(const cv::Mat1f& pixels, const cv::Mat1b& mask, cv::Mat3f& voxels) const
{
	Eigen::Vector4d epix;
	Eigen::Vector4d evox;

	epix(3) = 1; // w does not change.

	for (int r = 0; r < pixels.rows; ++r)
	{
		const float* pixels_data = pixels.ptr<float>(r);
		const uchar* mask_data = mask.ptr<uchar>(r);
		Vec3f* voxels_data = voxels.ptr<Vec3f>(r);
		for (int c = 0; c < pixels.cols; ++c)
		{
			if (!mask_data[c])
				continue;
			const float d = pixels_data[c];
			epix(0) = c*d;
			epix(1) = r*d;
			epix(2) = d;
			evox = impl->inv_project_transform * epix;
			voxels_data[c][0] = evox(0);
			voxels_data[c][1] = evox(1);
			voxels_data[c][2] = evox(2);
		}
	}
}


cv::Point3f Pose3D :: unprojectFromImage(const cv::Point2f& p, double depth) const
{
	Eigen::Vector4d ep (p.x, p.y, depth, 1);
	Eigen::Vector4d output;
	impl->unprojectFromImage(ep, output);
	return toVec3f(output);
}


/// KDuncan  - Addition
cv::Point3f Pose3D :: unprojectFromKinectImage(const cv::Point2f& p, double depth) const
{
	Eigen::Vector4d ep (p.x, p.y, depth, 1);
	Eigen::Vector4d output;
	impl->unprojectFromKinectImage(ep, output);
	return toVec3f(output);
}


void Pose3D :: applyTransformBefore(const Pose3D& rhs_pose)
{
	// impl->camera_transform = impl->camera_transform * rhs_pose.impl->camera_transform;
	// impl->computeProjectiveTransform();

	// First extracting translation and rotation components to avoid the cumulation
	// of numerical errors, eventually leading to invalid transformation matrices,
	// e.g. with < 1 determinant.
	applyTransformBefore(rhs_pose.cvTranslation(), rhs_pose.cvEulerRotation());
}


void Pose3D :: applyTransformAfter(const Pose3D& rhs_pose)
{
	// impl->camera_transform = rhs_pose.impl->camera_transform * impl->camera_transform;
	// impl->computeProjectiveTransform();

	// First extracting translation and rotation components to avoid the cumulation
	// of numerical errors, eventually leading to invalid transformation matrices,
	// e.g. with < 1 determinant.
	applyTransformAfter(rhs_pose.cvTranslation(), rhs_pose.cvEulerRotation());
}


void Pose3D :: applyTransformAfter(const cv::Vec3f& translation, const cv::Mat1d& rotation_matrix)
{
	Eigen::Matrix3d emat; toEigen(rotation_matrix, emat);
	impl->camera_transform.translate(toEigenVector3d(translation));
	impl->camera_transform.rotate(emat);
	impl->computeProjectiveTransform();
}


void Pose3D :: applyTransformBefore(const cv::Vec3f& translation, const cv::Mat1d& rotation_matrix)
{
	Eigen::Matrix3d emat; toEigen(rotation_matrix, emat);
	impl->camera_transform.prerotate(emat);
	impl->camera_transform.pretranslate(toEigenVector3d(translation));
	impl->computeProjectiveTransform();
}


void Pose3D :: applyTransformBefore(const cv::Vec3f& translation, const cv::Vec3f& rotation_euler_angles)
{
	impl->camera_transform.translate(toEigenVector3d(translation));
	impl->camera_transform.rotate(Eigen::AngleAxisd(rotation_euler_angles[2], Eigen::Vector3d::UnitZ())
		* Eigen::AngleAxisd(rotation_euler_angles[1], Eigen::Vector3d::UnitY())
		* Eigen::AngleAxisd(rotation_euler_angles[0], Eigen::Vector3d::UnitX()));
	impl->computeProjectiveTransform();
}


void Pose3D :: applyTransformAfter(const cv::Vec3f& translation, const cv::Vec3f& rotation_euler_angles)
{
	impl->camera_transform.prerotate(Eigen::AngleAxisd(rotation_euler_angles[2], Eigen::Vector3d::UnitZ())
		* Eigen::AngleAxisd(rotation_euler_angles[1], Eigen::Vector3d::UnitY())
		* Eigen::AngleAxisd(rotation_euler_angles[0], Eigen::Vector3d::UnitX()));
	impl->camera_transform.pretranslate(toEigenVector3d(translation));
	impl->computeProjectiveTransform();
}




} /* ocl */