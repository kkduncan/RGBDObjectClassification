#pragma once
#ifndef OCL_POSE_3D_H_
#define OCL_POSE_3D_H_

#include <opencv2/core/core.hpp>

namespace ocl {

/**
 * Some utility functions
 */

# define for_all_rc(im) \
	for (int r = 0; r < (im).rows; ++r) \
	for (int c = 0; c < (im).cols; ++c)


template <typename EScalarType>
inline cv::Vec3f toVec3f(const Eigen::Matrix<EScalarType,3,1>& v)
{
	return cv::Vec3f(v(0),v(1),v(2));
}	


template <typename EScalarType>
inline cv::Vec3f toVec3f(const Eigen::Matrix<EScalarType,4,1>& v)
{
	return cv::Vec3f(v(0),v(1),v(2));
}


template <typename CvScalarType, typename EScalarType>
inline void toEigen(const cv::Point3_<CvScalarType>& p, Eigen::Matrix<EScalarType,3,1>& ep)
{
	ep(0) = p.x;
	ep(1) = p.y;
	ep(2) = p.z;
}


template <typename CvScalarType, typename EScalarType>
inline void toEigen(const cv::Vec<CvScalarType,3>& p, Eigen::Matrix<EScalarType,3,1>& ep)
{
	ep(0) = p[0];
	ep(1) = p[1];
	ep(2) = p[2];
}


#ifdef _MSC_VER
inline void toEigen(const cv::Mat1d& mat, Eigen::Matrix3d& ep)
{
	for (int r = 0; r < 3; ++r)
		for (int c = 0; c < 3; ++c)
			ep(r,c) = mat(r,c);
}
#else
template <typename CvScalarType, typename EScalarType, int H, int W>
inline void toEigen(const cv::Mat_<CvScalarType>& mat, Eigen::Matrix<EScalarType,H,W>& ep)
{
	for (int r = 0; r < H; ++r)
		for (int c = 0; c < W; ++c)
			ep(r,c) = mat(r,c);
}
#endif


inline void toOpencv(const Eigen::Matrix4d& ep,
	cv::Mat1f& mat)
{
	for (int r = 0; r < 4; ++r)
		for (int c = 0; c < 4; ++c)
			mat(r,c) = ep(r,c);
}


inline void toOpencv(const Eigen::Matrix4d& ep,
	cv::Mat1d& mat)
{
	for (int r = 0; r < 4; ++r)
		for (int c = 0; c < 4; ++c)
			mat(r,c) = ep(r,c);
}


template <class ScalarType>
cv::Vec3d toVec3d(const cv::Mat_<ScalarType>& m)
{
	if (m.rows == 3 && m.cols == 1) {
		printf("m is not a vector.");
	}
	return cv::Vec3d(m(0,0), m(1,0), m(2,0));
}


inline Eigen::Vector3d toEigenVector3d(const cv::Vec3f& v)
{ 
	Eigen::Vector3d r; 
	toEigen(v, r); 
	return r; 
}


template <typename CvScalarType, typename EScalarType>
inline void toEigen(const cv::Point3_<CvScalarType>& p, Eigen::Matrix<EScalarType,4,1>& ep)
{
	ep(0) = p.x;
	ep(1) = p.y;
	ep(2) = p.z;
	ep(3) = 1;
}


inline float         sqr(float x)         { return x*x; }
inline double        sqr(double x)        { return x*x; }



/**
* Represent transformations within a Pin-Hole camera model
* Can be used both for 3D camera transforms and projections
* on an image plane using perspective or othogonal model.
* The default representation is OpenGL-like:
*   Y
*   |__ X
*  /
* Z
*
* This code has been adopted from nestk's version of Pose3D
*/
class Pose3D
{
private:
	class PrivatePose3D;
	friend class PrivatePose3D;
	PrivatePose3D* impl;

public:
	Pose3D();
	~Pose3D();

	Pose3D(const Pose3D& rhs);
	Pose3D& operator=(const Pose3D& rhs);

	/*! Set parameters from intrinsics matrix. */
	void setCameraParametersFromOpencv(const cv::Mat1d& cv_matrix);

	/*!  Set parameters from intrinsics matrix. */
	void setCameraParameters(double fx, double fy, double cx, double cy);

	/*!
	 * Transform a right camera into a left camera using stereo parameters.
	 * \param intrinsics_matrix intrinsics matrix of the left camera.
	 * \param R extrinsics 3x3 rotation matrix.
	 * \param T extrinsics 1x3 translation matrix.
	 */
	void toLeftCamera(const cv::Mat1d& intrinsics_matrix,
		const cv::Mat1d& R,
		const cv::Mat1d& T);

	/*!
	* Transform a left camera into a right camera using stereo parameters.
	* @see toLeftCamera
	*/
	void toRightCamera(const cv::Mat1d& cv_matrix,
		const cv::Mat1d& R,
		const cv::Mat1d& T);

	/*! Focal lenghts in pixels. */
	double focalX() const { return m_focal_x; }
	double focalY() const { return m_focal_y; }

	/*! Image plane center. */
	double imageCenterX() const { return m_image_center_x; }
	double imageCenterY() const { return m_image_center_y; }

	/*! Mean focal. */
	double meanFocal() const { return (m_focal_x + m_focal_y)/2.0; }

	/*! Whether it can be used as a complete camera model. */
	bool isValid() const { return m_has_camera_params; }

	/*! Return the determinant of the projection matrix. */
	float determinant() const;


public:
	/*! Returns the camera translation as OpenCV 3x3 matrix. */
	const cv::Vec3f cvTranslation() const;

	/*!
	* Returns the camera rotation as OpenCV vector of euler angles.
	* First angle is around X, second around Y and third around Z.
	*/
	const cv::Vec3f cvEulerRotation() const;

	/*! Returns the rotation as a quaternion. */
	cv::Vec4f cvQuaternionRotation() const;

	/*! Returns the camera transform as an OpenCV float 4x4 matrix. */
	const cv::Mat1f cvCameraTransform() const;

	/*! Returns the camera transform as an OpenCV double 4x4 matrix. */
	const cv::Mat1d cvCameraTransformd() const;

	/*!
	* Returns the camera transform as an OpenCV 3x3 rotation matrix
	* and a translation vector as a 3x1 matrix.
	* Useful to update calibration parameters.
	*/
	void cvRotationMatrixTranslation(cv::Mat1d& translation, cv::Mat1d& rotation) const;

	/*! Returns the inverse camera transform as an OpenCV 4x4 matrix. */
	cv::Mat1f cvInvCameraTransform() const;

	/*! Returns the 4x4 projection matrix (intrinsics * camera) */
	cv::Mat1f cvProjectionMatrix() const;

	/*! Returns the 4x4 inverse projection matrix from image to plane. */
	cv::Mat1f cvInvProjectionMatrix() const;

	/*! Reset the camera transform to Identity. */
	void resetCameraTransform();

	/*! Invert the camera transform. */
	void invert();

	/*! Return the inverted transform. */
	Pose3D inverted() const;

	/*!
	* Set the 3D camera transformation from OpenCV translation
	* and rodrigues vector.
	*/
	void setCameraTransform(const cv::Mat1d& tvec, const cv::Mat1d& rvec);

	/*! Set the 3D camera transform from 4x4 matrix. */
	void setCameraTransform(const cv::Mat1f& H);

	/*! Set the 3D camera transform from 4x4 matrix. */
	void setCameraTransform(const cv::Mat1d& H);

	/*! Set the 3D camera transform from 3x3 fundamental matrix. */
	void setCameraTransformFromCvFundamentalMatrix(const cv::Mat1f& F);

	/*! Apply a camera transform on the right. */
	void applyTransformBefore(const Pose3D& rhs_pose);
	void applyTransformBefore(const cv::Vec3f& cvTranslation, const cv::Vec3f& rotation_euler_angles);
	void applyTransformBefore(const cv::Vec3f& cvTranslation, const cv::Mat1d& rotation_matrix);

	/*! Apply a camera transform on the left. */
	void applyTransformAfter(const Pose3D& rhs_pose);
	void applyTransformAfter(const cv::Vec3f& translation, const cv::Vec3f& rotation_euler_angles);
	void applyTransformAfter(const cv::Vec3f& translation, const cv::Mat1d& rotation_matrix);

public:
	/*! Apply the camera transform on a given 3D point. */
	cv::Point3f cameraTransform(const cv::Point3f& p) const;

	/*! Apply the inverse camera transform on a given 3D point. */
	cv::Point3f invCameraTransform(const cv::Point3f& p) const;

	/*! Project a 3D point to image plane. */
	cv::Point3f projectToImage(const cv::Point3f& p) const;

	/*! Project a set of 3D points onto image plane. */
	void projectToImage(const cv::Mat3f& voxels, const cv::Mat1b& mask, cv::Mat3f& pixels) const;

	/*! Project a point from image plane to 3D using the given depth. */
	cv::Point3f unprojectFromImage(const cv::Point2f& p, double depth) const;
	/*! KDuncan - additions */
	cv::Point3f unprojectFromKinectImage(const cv::Point2f& p, double depth) const;

	cv::Point3f unprojectFromImage(const cv::Point3f& p) const
	{ return unprojectFromImage(cv::Point2f(p.x,p.y), p.z); }

	/*! Project a set of image points to 3D. */
	void unprojectFromImage(const cv::Mat1f& pixels, const cv::Mat1b& mask, cv::Mat3f& voxels) const;

private:
	double m_focal_x;
	double m_focal_y;
	double m_image_center_x;
	double m_image_center_y;
	bool m_has_camera_params;
};



} /* ocl */


#endif /* OCL_POSE_3D_H_ */
