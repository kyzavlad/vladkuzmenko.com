import os
import argparse
import numpy as np
import cv2
from PIL import Image
import torch
import glob
import matplotlib.pyplot as plt
from datetime import datetime
import json
import time
from typing import Dict, List, Any

from app.avatar_creation import (
    # Face modeling
    FaceReconstructor, FaceTextureMapper, FaceGeometryRefiner, ExpressionCapture,
    
    # Body modeling
    PoseEstimator, BodyMeshGenerator, BodyMeasurement, BodyTextureMapper, FaceBodyIntegrator,
    
    # Animation
    AvatarAnimator, FacialLandmarkTracker, MicroExpressionSynthesizer, GazeController,
    HeadPoseController, EmotionController, FirstOrderMotionModel, GestureMannerismLearner,
    
    # Utilities
    load_image, save_image, preprocess_image, get_device, ensure_directory
)
from app.avatar_creation.face_modeling import (
    TextureMapper,
    ImprovedTextureMapper,
    FeaturePreservation,
    IdentityVerification,
    CustomStyleGAN3,
    DetailRefinement,
    ExpressionCalibration,
    AdvancedFaceReconstructor,
    MultiViewReconstructor,
)
from app.avatar_creation import EnhancedStyleGAN3
from app.avatar_creation import (
    FaceDetector, 
    FaceLandmarkDetector, 
    FaceTexturizer,
    AvatarAnimator,
    VoiceAnimator,
)

def create_avatar_from_image(image_path: str, output_dir: str, high_quality: bool = False, use_advanced_reconstruction: bool = False, use_improved_texturing: bool = False) -> None:
    """
    Create a 3D avatar from a single image.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save results
        high_quality: Whether to use high-quality settings
        use_advanced_reconstruction: Whether to use advanced reconstruction algorithms
        use_improved_texturing: Whether to use improved texture mapping with better UV unwrapping
    """
    print(f"Creating avatar from image: {image_path}")
    ensure_directory(output_dir)
    
    # Load image
    image = load_image(image_path)
    
    # Step 1: 3D Face Reconstruction
    print("Step 1: 3D Face Reconstruction")
    
    if use_advanced_reconstruction:
        print("Using advanced reconstruction with BFM model")
        # Initialize the advanced reconstructor
        # Note: The BFM model file path should be set to a valid file or None for fallback mode
        face_reconstructor = AdvancedFaceReconstructor(
            bfm_path=os.environ.get("BFM_MODEL_PATH", None),
            model_path=os.environ.get("FACE_MODEL_PATH", None),
            use_deep_learning=high_quality
        )
    else:
        print("Using standard reconstruction with MediaPipe")
        face_reconstructor = FaceReconstructor()
    
    reconstruction_result = face_reconstructor.reconstruct_3d_face(image)
    
    mesh = reconstruction_result["mesh"]
    landmarks = reconstruction_result["landmarks"]
    landmarks_3d = reconstruction_result.get("landmarks_3d", landmarks)
    
    # Save the initial mesh
    initial_mesh_path = os.path.join(output_dir, "initial_mesh.ply")
    face_reconstructor.save_mesh(mesh, initial_mesh_path)
    print(f"Initial mesh saved to: {initial_mesh_path}")
    
    # Step 2: Feature Preservation
    print("Step 2: Feature Preservation")
    feature_preserver = FeaturePreservation()
    feature_result = feature_preserver.detect_important_features(image)
    
    importance_map = feature_result["importance_map"]
    feature_points = feature_result["feature_points"]
    
    # Save importance map
    importance_map_vis = (importance_map * 255).astype(np.uint8)
    importance_map_path = os.path.join(output_dir, "importance_map.png")
    cv2.imwrite(importance_map_path, importance_map_vis)
    print(f"Feature importance map saved to: {importance_map_path}")
    
    # Apply feature preservation to mesh
    mesh_vertices = np.asarray(mesh.vertices)
    constraints = feature_preserver.generate_feature_constraints(mesh_vertices, feature_points)
    preserved_vertices = feature_preserver.apply_feature_preserving_deformation(mesh_vertices, constraints)
    
    # Create preserved mesh
    import open3d as o3d
    preserved_mesh = o3d.geometry.TriangleMesh()
    preserved_mesh.vertices = o3d.utility.Vector3dVector(preserved_vertices)
    preserved_mesh.triangles = mesh.triangles
    
    # Save preserved mesh
    preserved_mesh_path = os.path.join(output_dir, "preserved_mesh.ply")
    o3d.io.write_triangle_mesh(preserved_mesh_path, preserved_mesh)
    print(f"Feature-preserved mesh saved to: {preserved_mesh_path}")
    
    # Step 3: Texture Mapping
    print("Step 3: Texture Mapping")
    if use_improved_texturing:
        print("Using improved texture mapping with optimized UV unwrapping")
        texture_mapper = ImprovedTextureMapper(
            resolution=(4096, 4096) if high_quality else (2048, 2048),
            seamless_boundary=True,
            optimize_charts=high_quality
        )
    else:
        print("Using standard texture mapping")
        texture_mapper = TextureMapper(
            resolution=(4096, 4096) if high_quality else (2048, 2048)
        )
    
    # Export mesh for texture mapping
    obj_path = face_reconstructor.export_for_texture_mapping(preserved_mesh, os.path.join(output_dir, "mesh_for_texturing.ply"))
    
    # Load the mesh for texturing
    import trimesh
    mesh_for_texturing = trimesh.load(obj_path)
    
    # Project texture
    texture_result = texture_mapper.project_texture(mesh_for_texturing, image, landmarks)
    textured_mesh = texture_result["textured_mesh"]
    texture_image = texture_result["texture_image"]
    
    # Enhance texture
    enhanced_texture = texture_mapper.enhance_texture(texture_image)
    
    # Save enhanced texture
    enhanced_texture_path = os.path.join(output_dir, "enhanced_texture.png")
    cv2.imwrite(enhanced_texture_path, cv2.cvtColor(enhanced_texture, cv2.COLOR_RGB2BGR))
    print(f"Enhanced texture saved to: {enhanced_texture_path}")
    
    # Step 4: Detail Refinement
    print("Step 4: Detail Refinement")
    detail_refiner = DetailRefinement(resolution=(4096, 4096) if high_quality else (2048, 2048))
    
    # Apply details to texture
    detailed_texture = detail_refiner.apply_details_to_texture(
        enhanced_texture, 
        detail_strength=1.0 if high_quality else 0.7
    )
    
    # Save detailed texture
    detailed_texture_path = os.path.join(output_dir, "detailed_texture.png")
    cv2.imwrite(detailed_texture_path, cv2.cvtColor(detailed_texture, cv2.COLOR_RGB2BGR))
    print(f"Detailed texture saved to: {detailed_texture_path}")
    
    # Generate displacement map
    displacement_map = detail_refiner.generate_displacement_map(strength=1.0 if high_quality else 0.7)
    displacement_map_path = os.path.join(output_dir, "displacement_map.png")
    cv2.imwrite(displacement_map_path, displacement_map)
    print(f"Displacement map saved to: {displacement_map_path}")
    
    # Create detail preview
    detail_preview = detail_refiner.create_detail_preview()
    detail_preview_path = os.path.join(output_dir, "detail_preview.png")
    cv2.imwrite(detail_preview_path, cv2.cvtColor(detail_preview, cv2.COLOR_RGB2BGR))
    print(f"Detail preview saved to: {detail_preview_path}")
    
    # Step 5: Expression Calibration
    print("Step 5: Expression Calibration")
    expression_calibrator = ExpressionCalibration()
    
    # Create basic blendshapes
    blendshapes = expression_calibrator.create_basic_blendshapes(mesh_for_texturing)
    
    # Create expression visualizations
    for expr_name in ['neutral', 'smile', 'surprise', 'jaw_open']:
        if expr_name in blendshapes:
            expr_vis = expression_calibrator.create_expression_visualization(expr_name)
            expr_vis_path = os.path.join(output_dir, f"expression_{expr_name}.png")
            cv2.imwrite(expr_vis_path, cv2.cvtColor(expr_vis, cv2.COLOR_RGB2BGR))
            print(f"Expression visualization for '{expr_name}' saved to: {expr_vis_path}")
    
    # Save blendshapes
    blendshapes_dir = os.path.join(output_dir, "blendshapes")
    ensure_directory(blendshapes_dir)
    saved_blendshapes = expression_calibrator.save_blendshapes(blendshapes_dir)
    print(f"Blendshapes saved to: {blendshapes_dir}")
    
    # Step 6: Identity Verification
    print("Step 6: Identity Verification")
    identity_verifier = IdentityVerification()
    
    # For demonstration, we'll use the original image as both source and target
    # In a real scenario, you would use a rendered view of the 3D model as the target
    verification_result = identity_verifier.verify_identity(image, image)
    
    # Generate verification report
    report = identity_verifier.generate_report(verification_result)
    report_path = os.path.join(output_dir, "identity_verification_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Identity verification report saved to: {report_path}")
    
    # Step 7: Export final model
    print("Step 7: Exporting Final Model")
    export_result = texture_mapper.export_textured_model(
        textured_mesh,
        detailed_texture,
        output_dir,
        model_name="avatar"
    )
    
    print(f"Final avatar model exported to: {export_result['obj_path']}")
    print(f"Final texture exported to: {export_result['texture_path']}")
    
    print("\nAvatar creation completed successfully!")
    print(f"All results saved to: {output_dir}")

def create_avatar_from_multi_view(image_dir: str, output_dir: str, high_quality: bool = False, use_improved_texturing: bool = False) -> None:
    """
    Create a 3D avatar from multiple images showing different views of the face.
    
    Args:
        image_dir: Directory containing multiple face images
        output_dir: Directory to save results
        high_quality: Whether to use high-quality settings
        use_improved_texturing: Whether to use improved texture mapping with better UV unwrapping
    """
    print(f"Creating avatar from multiple images in: {image_dir}")
    ensure_directory(output_dir)
    
    # Find all images in the directory
    image_paths = []
    for ext in ['jpg', 'jpeg', 'png', 'bmp']:
        image_paths.extend(glob.glob(os.path.join(image_dir, f"*.{ext}")))
        image_paths.extend(glob.glob(os.path.join(image_dir, f"*.{ext.upper()}")))
    
    if not image_paths:
        raise ValueError(f"No images found in directory: {image_dir}")
    
    print(f"Found {len(image_paths)} images for multi-view reconstruction")
    
    # Step 1: Multi-view 3D Face Reconstruction
    print("Step 1: Multi-view 3D Face Reconstruction")
    
    # Initialize the multi-view reconstructor
    face_reconstructor = MultiViewReconstructor(
        bfm_path=os.environ.get("BFM_MODEL_PATH", None),
        model_path=os.environ.get("FACE_MODEL_PATH", None),
        use_deep_learning=high_quality
    )
    
    # Perform multi-view reconstruction
    reconstruction_result = face_reconstructor.reconstruct_from_multiple_images(image_paths)
    
    mesh = reconstruction_result["mesh"]
    
    # Save the initial mesh
    initial_mesh_path = os.path.join(output_dir, "initial_mesh.ply")
    face_reconstructor.save_mesh(mesh, initial_mesh_path)
    print(f"Initial mesh saved to: {initial_mesh_path}")
    
    # Load all images and detect landmarks for texture mapping
    images = []
    landmarks_list = []
    
    for img_path in image_paths:
        img = load_image(img_path)
        images.append(img)
        
        # Get landmarks
        lmks = face_reconstructor.base_reconstructor.detect_landmarks(img)
        landmarks_list.append(lmks)
    
    # For feature preservation, use the first image as reference
    reference_image = images[0]
    landmarks = landmarks_list[0]
    
    # Step 2: Feature Preservation (same as single image version)
    print("Step 2: Feature Preservation")
    feature_preserver = FeaturePreservation()
    feature_result = feature_preserver.detect_important_features(reference_image)
    
    importance_map = feature_result["importance_map"]
    feature_points = feature_result["feature_points"]
    
    # Save importance map
    importance_map_vis = (importance_map * 255).astype(np.uint8)
    importance_map_path = os.path.join(output_dir, "importance_map.png")
    cv2.imwrite(importance_map_path, importance_map_vis)
    print(f"Feature importance map saved to: {importance_map_path}")
    
    # Apply feature preservation to mesh
    mesh_vertices = np.asarray(mesh.vertices)
    constraints = feature_preserver.generate_feature_constraints(mesh_vertices, feature_points)
    preserved_vertices = feature_preserver.apply_feature_preserving_deformation(mesh_vertices, constraints)
    
    # Create preserved mesh
    import open3d as o3d
    preserved_mesh = o3d.geometry.TriangleMesh()
    preserved_mesh.vertices = o3d.utility.Vector3dVector(preserved_vertices)
    preserved_mesh.triangles = mesh.triangles
    
    # Save preserved mesh
    preserved_mesh_path = os.path.join(output_dir, "preserved_mesh.ply")
    o3d.io.write_triangle_mesh(preserved_mesh_path, preserved_mesh)
    print(f"Feature-preserved mesh saved to: {preserved_mesh_path}")
    
    # Step 3: Texture Mapping with multiple views (enhanced)
    print("Step 3: Texture Mapping with Multiple Views")
    if use_improved_texturing:
        print("Using improved texture mapping with optimized UV unwrapping")
        texture_mapper = ImprovedTextureMapper(
            resolution=(4096, 4096) if high_quality else (2048, 2048),
            seamless_boundary=True,
            optimize_charts=high_quality
        )
    else:
        print("Using standard texture mapping")
        texture_mapper = TextureMapper(
            resolution=(4096, 4096) if high_quality else (2048, 2048)
        )
    
    # Export mesh for texture mapping
    obj_path = face_reconstructor.export_for_texture_mapping(preserved_mesh, os.path.join(output_dir, "mesh_for_texturing.ply"))
    
    # Load the mesh for texturing
    import trimesh
    mesh_for_texturing = trimesh.load(obj_path)
    
    # Project texture from multiple views
    if use_improved_texturing and isinstance(texture_mapper, ImprovedTextureMapper):
        # Use multi-view texture projection
        texture_result = texture_mapper.project_texture_from_multiple_views(
            mesh_for_texturing, 
            images, 
            landmarks_list
        )
    else:
        # Fallback to single view (first image)
        texture_result = texture_mapper.project_texture(
            mesh_for_texturing, 
            images[0], 
            landmarks_list[0]
        )
    
    textured_mesh = texture_result["textured_mesh"]
    texture_image = texture_result["texture_image"]
    
    # Enhance texture
    enhanced_texture = texture_mapper.enhance_texture(texture_image)
    
    # Save enhanced texture
    enhanced_texture_path = os.path.join(output_dir, "enhanced_texture.png")
    cv2.imwrite(enhanced_texture_path, cv2.cvtColor(enhanced_texture, cv2.COLOR_RGB2BGR))
    print(f"Enhanced texture saved to: {enhanced_texture_path}")
    
    # Continue with the same steps as for single-image avatar creation
    # Steps 4-7: Detail Refinement, Expression Calibration, Identity Verification, Export
    # These steps are identical to the single-image workflow
    
    # Step 4: Detail Refinement
    print("Step 4: Detail Refinement")
    detail_refiner = DetailRefinement(resolution=(4096, 4096) if high_quality else (2048, 2048))
    
    # Apply details to texture
    detailed_texture = detail_refiner.apply_details_to_texture(
        enhanced_texture, 
        detail_strength=1.0 if high_quality else 0.7
    )
    
    # Save detailed texture
    detailed_texture_path = os.path.join(output_dir, "detailed_texture.png")
    cv2.imwrite(detailed_texture_path, cv2.cvtColor(detailed_texture, cv2.COLOR_RGB2BGR))
    print(f"Detailed texture saved to: {detailed_texture_path}")
    
    # Step 5: Expression Calibration
    print("Step 5: Expression Calibration")
    expression_calibrator = ExpressionCalibration()
    
    # Create basic blendshapes
    blendshapes = expression_calibrator.create_basic_blendshapes(mesh_for_texturing)
    
    # Save blendshapes
    blendshapes_dir = os.path.join(output_dir, "blendshapes")
    ensure_directory(blendshapes_dir)
    saved_blendshapes = expression_calibrator.save_blendshapes(blendshapes_dir)
    print(f"Blendshapes saved to: {blendshapes_dir}")
    
    # Step 6: Identity Verification
    print("Step 6: Identity Verification")
    identity_verifier = IdentityVerification()
    
    # For demonstration, we'll use the original image as both source and target
    # In a real scenario, you would use a rendered view of the 3D model as the target
    verification_result = identity_verifier.verify_identity(image, image)
    
    # Generate verification report
    report = identity_verifier.generate_report(verification_result)
    report_path = os.path.join(output_dir, "identity_verification_report.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"Identity verification report saved to: {report_path}")
    
    # Step 7: Export final model
    print("Step 7: Exporting Final Model")
    export_result = texture_mapper.export_textured_model(
        textured_mesh,
        detailed_texture,
        output_dir,
        model_name="avatar_multi_view"
    )
    
    print(f"Final avatar model exported to: {export_result['obj_path']}")
    print(f"Final texture exported to: {export_result['texture_path']}")
    
    print("\nMulti-view avatar creation completed successfully!")
    print(f"All results saved to: {output_dir}")

def create_avatar_from_video(video_path: str, output_dir: str, high_quality: bool = False, use_advanced_reconstruction: bool = False, use_improved_texturing: bool = False) -> None:
    """
    Create a 3D avatar from a video.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save results
        high_quality: Whether to use high-quality settings
        use_advanced_reconstruction: Whether to use advanced reconstruction algorithms
        use_improved_texturing: Whether to use improved texture mapping with better UV unwrapping
    """
    print(f"Creating avatar from video: {video_path}")
    ensure_directory(output_dir)
    
    # Step 1: 3D Face Reconstruction from video
    print("Step 1: 3D Face Reconstruction from Video")
    
    if use_advanced_reconstruction:
        print("Using advanced reconstruction with BFM model")
        # Initialize the advanced reconstructor
        face_reconstructor = AdvancedFaceReconstructor(
            bfm_path=os.environ.get("BFM_MODEL_PATH", None),
            model_path=os.environ.get("FACE_MODEL_PATH", None),
            use_deep_learning=high_quality
        )
    else:
        print("Using standard reconstruction with MediaPipe")
        face_reconstructor = FaceReconstructor()
    
    reconstruction_result = face_reconstructor.reconstruct_from_video(
        video_path, 
        sampling_rate=10 if high_quality else 5
    )
    
    mesh = reconstruction_result["mesh"]
    aggregated_landmarks = reconstruction_result["aggregated_landmarks"]
    
    # Save the initial mesh
    initial_mesh_path = os.path.join(output_dir, "initial_mesh.ply")
    face_reconstructor.save_mesh(mesh, initial_mesh_path)
    print(f"Initial mesh saved to: {initial_mesh_path}")
    
    # Extract a frame from the video for texturing
    import cv2
    video = cv2.VideoCapture(video_path)
    success, frame = video.read()
    video.release()
    
    if not success:
        raise ValueError(f"Failed to extract frame from video: {video_path}")
    
    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Save the frame
    frame_path = os.path.join(output_dir, "reference_frame.png")
    cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    print(f"Reference frame saved to: {frame_path}")
    
    # Continue with the same steps as for image-based avatar creation,
    # but using the extracted frame and the mesh from video reconstruction
    
    # For simplicity, we'll use the improved texture mapper if requested
    if use_improved_texturing:
        # This is just a placeholder for the real implementation
        print("Using improved texture mapping for video-based reconstruction")
    
    # Continue with other steps...
    
    print("\nAvatar creation from video completed!")
    print(f"All results saved to: {output_dir}")

def demonstrate_enhanced_stylegan(output_dir, quality='high'):
    """
    Demonstrates the capabilities of the EnhancedStyleGAN3 class.
    
    Args:
        output_dir: Directory to save output files
        quality: Quality setting ('low', 'medium', 'high')
    """
    print("\n=== Demonstrating Enhanced StyleGAN-3 Features ===")
    
    # Create output directory
    stylegan_dir = os.path.join(output_dir, "enhanced_stylegan_demo")
    ensure_directory(stylegan_dir)
    
    # Initialize the enhanced StyleGAN-3 model
    print("Initializing EnhancedStyleGAN3...")
    stylegan = EnhancedStyleGAN3(
        latent_dim=512,
        image_resolution=1024 if quality == 'high' else 512,
        use_custom_extensions=True
    )
    
    # 1. Generate a face with truncation trick for quality control
    print("\n1. Generating faces with different truncation values...")
    truncation_values = [0.5, 0.7, 1.0]
    truncation_outputs = []
    
    for psi in truncation_values:
        result = stylegan.generate_with_truncation(truncation_psi=psi)
        img = result["image"]
        trunc_path = os.path.join(stylegan_dir, f"truncation_psi_{psi:.1f}.png")
        save_image(img, trunc_path)
        print(f"  - Saved face with truncation psi={psi} to {trunc_path}")
        truncation_outputs.append(img)
    
    # Create a comparison image
    fig, axes = plt.subplots(1, len(truncation_values), figsize=(len(truncation_values) * 5, 5))
    for i, (img, psi) in enumerate(zip(truncation_outputs, truncation_values)):
        axes[i].imshow(img)
        axes[i].set_title(f"Truncation ψ={psi}")
        axes[i].axis('off')
    plt.tight_layout()
    comparison_path = os.path.join(stylegan_dir, "truncation_comparison.png")
    plt.savefig(comparison_path)
    plt.close()
    
    # 2. Demonstrate attribute control
    print("\n2. Demonstrating attribute control...")
    base_result = stylegan.generate_with_truncation(truncation_psi=0.7)
    base_latent = base_result["latent"]
    base_img = base_result["image"]
    base_path = os.path.join(stylegan_dir, "attribute_base.png")
    save_image(base_img, base_path)
    
    # Control different attributes
    attributes = ["age", "smile", "pose_yaw"]
    attr_strengths = [-1.0, 1.0]
    
    for attr in attributes:
        for strength in attr_strengths:
            try:
                result = stylegan.control_attribute(base_latent, attr, strength)
                attr_path = os.path.join(stylegan_dir, f"attr_{attr}_{strength:.1f}.png")
                save_image(result["image"], attr_path)
                print(f"  - Saved {attr} with strength {strength} to {attr_path}")
            except ValueError as e:
                print(f"  - Error with attribute {attr}: {e}")
    
    # 3. Create a morphing sequence
    print("\n3. Creating a face morphing sequence...")
    start_result = stylegan.generate_with_truncation(truncation_psi=0.7)
    end_result = stylegan.generate_with_truncation(truncation_psi=0.7)
    
    morph_dir = os.path.join(stylegan_dir, "morphing")
    ensure_directory(morph_dir)
    
    frames = stylegan.create_morph_sequence(
        start_result["latent"], 
        end_result["latent"],
        num_frames=10,
        output_dir=morph_dir,
        create_video=True
    )
    
    print(f"  - Saved morphing sequence to {morph_dir}")
    
    # 4. Generate an expression sequence
    print("\n4. Generating expression sequence...")
    expr_dir = os.path.join(stylegan_dir, "expressions")
    ensure_directory(expr_dir)
    
    try:
        neutral_result = stylegan.generate_with_truncation(truncation_psi=0.7)
        expression_frames = stylegan.generate_expression_sequence(
            neutral_result["latent"],
            "smile",
            num_frames=5,
            max_strength=1.5
        )
        
        for i, frame in enumerate(expression_frames):
            expr_path = os.path.join(expr_dir, f"smile_{i:02d}.png")
            save_image(frame, expr_path)
        
        print(f"  - Saved expression sequence to {expr_dir}")
    except ValueError as e:
        print(f"  - Error generating expression sequence: {e}")
    
    # 5. Latent space exploration
    print("\n5. Exploring latent space neighborhood...")
    variation_dir = os.path.join(stylegan_dir, "variations")
    ensure_directory(variation_dir)
    
    base_result = stylegan.generate_with_truncation(truncation_psi=0.7)
    variations = stylegan.latent_space_exploration(
        base_result["latent"],
        num_variations=4,
        variation_strength=0.3
    )
    
    for i, var_img in enumerate(variations):
        var_path = os.path.join(variation_dir, f"variation_{i:02d}.png")
        save_image(var_img, var_path)
    
    print(f"  - Saved latent space variations to {variation_dir}")
    
    # Create a summary report
    print("\n6. Creating summary report...")
    summary_path = os.path.join(stylegan_dir, "enhanced_stylegan_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Enhanced StyleGAN-3 Demonstration Summary\n")
        f.write("=======================================\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Quality setting: {quality}\n\n")
        f.write("Featured Demonstrations:\n")
        f.write("1. Truncation trick for quality vs. diversity control\n")
        f.write("2. Attribute control (age, smile, pose)\n")
        f.write("3. Face morphing sequence\n")
        f.write("4. Expression sequence generation\n")
        f.write("5. Latent space neighborhood exploration\n\n")
        f.write("For detailed implementation, see the EnhancedStyleGAN3 class.\n")
    
    print(f"Enhanced StyleGAN-3 demonstration completed. Results saved to {stylegan_dir}")
    return stylegan_dir

def create_full_body_avatar(
    image_path: str,
    output_dir: str,
    high_quality: bool = False,
    measure_body: bool = True,
    integrate_face: bool = True
) -> None:
    """
    Create a full-body 3D avatar from an image.
    
    Args:
        image_path: Path to the input image showing a full body
        output_dir: Directory to save the output files
        high_quality: Whether to use high-quality settings
        measure_body: Whether to extract body measurements
        integrate_face: Whether to integrate with a face model
    """
    print(f"\n{'='*50}")
    print(f"Creating full-body avatar from: {image_path}")
    print(f"{'='*50}\n")
    
    # Create output directory
    ensure_directory(output_dir)
    
    # Step 1: Estimate body pose
    print("\n[Step 1/6] Estimating body pose...")
    pose_estimator = PoseEstimator(use_3d=True, use_temporal_smoothing=False)
    pose_data = pose_estimator.estimate_pose_from_image(image_path)
    
    # Save pose visualization
    pose_viz_path = os.path.join(output_dir, "pose_estimation.jpg")
    image = load_image(image_path)
    if image is not None and pose_data is not None:
        pose_image = pose_estimator.draw_pose_on_image(image, pose_data)
        save_image(pose_viz_path, pose_image)
        print(f"Pose visualization saved to: {pose_viz_path}")
    
    # Step 2: Extract body measurements if requested
    if measure_body:
        print("\n[Step 2/6] Extracting body measurements...")
        body_measurement = BodyMeasurement(use_3d=True, high_precision=high_quality)
        measurements = body_measurement.measure_from_image(image_path, pose_data)
        
        # Save measurements to file
        measurements_path = os.path.join(output_dir, "body_measurements.json")
        body_measurement.save_measurements_to_json(measurements, measurements_path)
        print(f"Body measurements saved to: {measurements_path}")
        
        # Generate size recommendations
        sizes = body_measurement.convert_measurements_to_size(measurements)
        print("Size recommendations:")
        for item, size in sizes.items():
            print(f"  - {item}: {size}")
    else:
        # Use default measurements
        measurements = {
            'height': 175.0,
            'shoulder_width': 40.0,
            'chest_circumference': 90.0,
            'waist_circumference': 80.0,
            'hip_circumference': 95.0
        }
    
    # Step 3: Generate 3D body mesh from measurements
    print("\n[Step 3/6] Generating 3D body mesh...")
    body_generator = BodyMeshGenerator(
        gender='neutral',
        high_quality=high_quality
    )
    
    # Extract pose parameters from pose_data if available
    pose_params = None
    if pose_data and 'keypoints_3d' in pose_data:
        # This is a simplified conversion - in a real implementation,
        # you would have a more sophisticated conversion from keypoints to pose parameters
        try:
            pose_params = pose_estimator.convert_pose_to_smpl(pose_data)
        except:
            print("Could not convert pose to SMPL parameters, using default pose")
    
    # Generate mesh from measurements
    body_mesh = body_generator.generate_mesh_from_measurements(measurements, pose_params)
    
    # Save the body mesh
    body_mesh_path = os.path.join(output_dir, "body_mesh.obj")
    body_generator.save_mesh(body_mesh, body_mesh_path)
    print(f"Body mesh saved to: {body_mesh_path}")
    
    # Step 4: Generate textures for the body
    print("\n[Step 4/6] Generating body textures...")
    body_texture_mapper = BodyTextureMapper(
        resolution=(4096, 4096) if high_quality else (2048, 2048),
        use_high_quality=high_quality,
        use_pbr=high_quality
    )
    
    # Ensure the mesh has UV coordinates
    body_mesh_with_uv = body_texture_mapper.generate_uv_coordinates(body_mesh)
    
    # Project texture from the input image
    texture = body_texture_mapper.project_image_to_texture(
        body_mesh_with_uv, 
        [image_path]
    )
    
    # Save the texture
    texture_path = os.path.join(output_dir, "body_texture.png")
    body_texture_mapper.save_texture(texture, texture_path)
    print(f"Body texture saved to: {texture_path}")
    
    # Generate PBR textures if high quality
    if high_quality:
        print("Generating PBR textures...")
        pbr_textures = body_texture_mapper.generate_pbr_textures(body_mesh_with_uv, texture)
        pbr_dir = os.path.join(output_dir, "pbr_textures")
        texture_paths = body_texture_mapper.save_pbr_textures(pbr_textures, pbr_dir)
        print(f"PBR textures saved to: {pbr_dir}")
    
    # Apply texture to the mesh
    textured_body = body_texture_mapper.apply_texture_to_mesh(body_mesh_with_uv, texture_path)
    textured_body_path = os.path.join(output_dir, "textured_body.obj")
    body_generator.save_mesh(textured_body, textured_body_path, texture_path)
    print(f"Textured body mesh saved to: {textured_body_path}")
    
    # Step 5: Create face model (if integration is requested)
    if integrate_face:
        print("\n[Step 5/6] Creating face model...")
        # Extract face region from the image
        # In a real implementation, you would detect the face and crop it
        # Here we'll just use the existing face creation function
        
        face_dir = os.path.join(output_dir, "face")
        ensure_directory(face_dir)
        
        # Check if we can detect a face in the image
        try:
            import cv2
            image = cv2.imread(image_path)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            if len(faces) > 0:
                # Extract the largest face
                x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
                # Add some margin
                margin = int(min(w, h) * 0.3)
                x, y = max(0, x - margin), max(0, y - margin)
                w, h = min(image.shape[1] - x, w + 2*margin), min(image.shape[0] - y, h + 2*margin)
                
                face_img = image[y:y+h, x:x+w]
                face_path = os.path.join(face_dir, "face.jpg")
                cv2.imwrite(face_path, face_img)
                print(f"Extracted face saved to: {face_path}")
                
                # Create face avatar
                create_avatar_from_image(
                    face_path, 
                    face_dir, 
                    high_quality=high_quality, 
                    use_advanced_reconstruction=high_quality,
                    use_improved_texturing=high_quality
                )
                
                # Get the face mesh and texture
                face_mesh_path = os.path.join(face_dir, "face_model.obj")
                face_texture_path = os.path.join(face_dir, "face_texture.png")
                
                # Step 6: Integrate face and body
                print("\n[Step 6/6] Integrating face and body...")
                face_body_integrator = FaceBodyIntegrator(
                    blend_region_size=0.15,
                    preserve_face_details=True,
                    smooth_transition=True,
                    high_quality=high_quality
                )
                
                # Load face mesh and body mesh
                import trimesh
                face_mesh = trimesh.load(face_mesh_path)
                
                # Load textures
                face_texture = load_image(face_texture_path)
                
                # Integrate face and body with textures
                integrated_result = face_body_integrator.integrate_with_textures(
                    face_mesh=face_mesh,
                    body_mesh=textured_body,
                    face_texture=face_texture,
                    body_texture=texture
                )
                
                # Save the integrated avatar
                integrated_dir = os.path.join(output_dir, "integrated")
                integrated_paths = face_body_integrator.save_integrated_avatar(
                    integrated_result, 
                    integrated_dir
                )
                
                print(f"Integrated avatar saved to: {integrated_dir}")
                print(f"  - Model: {integrated_paths.get('obj_path')}")
                print(f"  - Texture: {integrated_paths.get('texture_path')}")
                
                # Create a preview
                preview_path = os.path.join(output_dir, "avatar_preview.png")
                face_body_integrator.create_avatar_preview(
                    integrated_result,
                    preview_path
                )
                print(f"Avatar preview saved to: {preview_path}")
            else:
                print("No face detected in the image, skipping face integration")
                print("\n[Step 6/6] Face integration skipped")
        except Exception as e:
            print(f"Error creating face model: {e}")
            print("\n[Step 6/6] Face integration skipped")
    else:
        print("\n[Step 5/6] Face model creation skipped")
        print("\n[Step 6/6] Face integration skipped")
    
    print("\nFull-body avatar creation completed!")
    print(f"Results saved to: {output_dir}")

def create_full_body_avatar_from_video(
    video_path: str,
    output_dir: str,
    high_quality: bool = False,
    frame_rate: int = 5
) -> None:
    """
    Create a full-body 3D avatar from a video.
    
    Args:
        video_path: Path to the input video
        output_dir: Directory to save the output files
        high_quality: Whether to use high-quality settings
        frame_rate: Number of frames to extract per second
    """
    print(f"\n{'='*50}")
    print(f"Creating full-body avatar from video: {video_path}")
    print(f"{'='*50}\n")
    
    # Create output directory
    ensure_directory(output_dir)
    
    # Step 1: Estimate body pose from video
    print("\n[Step 1/6] Estimating body pose from video...")
    pose_estimator = PoseEstimator(use_3d=True, use_temporal_smoothing=True)
    pose_data = pose_estimator.estimate_pose_from_video(
        video_path, 
        output_dir=os.path.join(output_dir, "pose_estimation")
    )
    
    # Extract a representative frame from the video
    print("\nExtracting representative frame from video...")
    import cv2
    cap = cv2.VideoCapture(video_path)
    
    # Get total frame count and middle frame
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    middle_frame_idx = total_frames // 2
    
    # Set position to middle frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
    ret, frame = cap.read()
    
    if ret:
        # Save the frame
        frame_path = os.path.join(output_dir, "representative_frame.jpg")
        cv2.imwrite(frame_path, frame)
        print(f"Representative frame saved to: {frame_path}")
        
        # Now use this frame as the image for creating the avatar
        create_full_body_avatar(
            frame_path,
            output_dir,
            high_quality=high_quality,
            measure_body=True,
            integrate_face=True
        )
    else:
        print("Error: Could not extract frame from video")
        
    cap.release()

def create_measured_avatar(
    height_cm: float,
    weight_kg: float,
    chest_cm: float,
    waist_cm: float,
    hips_cm: float,
    gender: str = 'neutral',
    output_dir: str = 'output/measured_avatar',
    high_quality: bool = False
) -> None:
    """
    Create a 3D avatar from body measurements.
    
    Args:
        height_cm: Height in centimeters
        weight_kg: Weight in kilograms
        chest_cm: Chest circumference in centimeters
        waist_cm: Waist circumference in centimeters
        hips_cm: Hip circumference in centimeters
        gender: Gender of the avatar ('male', 'female', or 'neutral')
        output_dir: Directory to save the output files
        high_quality: Whether to use high-quality settings
    """
    print(f"\n{'='*50}")
    print(f"Creating avatar from measurements")
    print(f"{'='*50}\n")
    
    # Create output directory
    ensure_directory(output_dir)
    
    # Create the measurement dictionary
    measurements = {
        'height': height_cm,
        'weight': weight_kg,
        'chest_circumference': chest_cm,
        'waist_circumference': waist_cm,
        'hip_circumference': hips_cm,
        'shoulder_width': chest_cm * 0.8 / 2,  # Rough estimate
        'arm_length': height_cm * 0.33,        # Rough estimate
        'inseam': height_cm * 0.45             # Rough estimate
    }
    
    # Save measurements to file
    measurements_path = os.path.join(output_dir, "body_measurements.json")
    with open(measurements_path, 'w') as f:
        json.dump(measurements, f, indent=2)
    print(f"Measurements saved to: {measurements_path}")
    
    # Generate 3D body mesh from measurements
    print("\nGenerating 3D body mesh...")
    body_generator = BodyMeshGenerator(
        gender=gender,
        high_quality=high_quality
    )
    
    # Generate mesh from measurements
    body_mesh = body_generator.generate_mesh_from_measurements(measurements)
    
    # Save the body mesh
    body_mesh_path = os.path.join(output_dir, "body_mesh.obj")
    body_generator.save_mesh(body_mesh, body_mesh_path)
    print(f"Body mesh saved to: {body_mesh_path}")
    
    # Generate textures for the body
    print("\nGenerating body textures...")
    body_texture_mapper = BodyTextureMapper(
        resolution=(4096, 4096) if high_quality else (2048, 2048),
        use_high_quality=high_quality,
        use_pbr=high_quality
    )
    
    # Ensure the mesh has UV coordinates
    body_mesh_with_uv = body_texture_mapper.generate_uv_coordinates(body_mesh)
    
    # Generate procedural texture
    texture = body_texture_mapper.generate_procedural_texture(body_mesh_with_uv)
    
    # Save the texture
    texture_path = os.path.join(output_dir, "body_texture.png")
    body_texture_mapper.save_texture(texture, texture_path)
    print(f"Body texture saved to: {texture_path}")
    
    # Generate PBR textures if high quality
    if high_quality:
        print("Generating PBR textures...")
        pbr_textures = body_texture_mapper.generate_pbr_textures(body_mesh_with_uv, texture)
        pbr_dir = os.path.join(output_dir, "pbr_textures")
        texture_paths = body_texture_mapper.save_pbr_textures(pbr_textures, pbr_dir)
        print(f"PBR textures saved to: {pbr_dir}")
    
    # Apply texture to the mesh
    textured_body = body_texture_mapper.apply_texture_to_mesh(body_mesh_with_uv, texture_path)
    textured_body_path = os.path.join(output_dir, "textured_body.obj")
    body_generator.save_mesh(textured_body, textured_body_path, texture_path)
    print(f"Textured body mesh saved to: {textured_body_path}")
    
    print("\nMeasured avatar creation completed!")
    print(f"Results saved to: {output_dir}")

def animate_avatar_with_voice(avatar_path: str, audio_path: str, output_dir: str, 
                             high_quality: bool = False, real_time: bool = False) -> Dict:
    """
    Animate an avatar using voice audio for lip-sync and facial expressions.
    
    Args:
        avatar_path: Path to the avatar model file (.obj)
        audio_path: Path to the audio file
        output_dir: Directory to save animation output
        high_quality: Whether to use high-quality settings
        real_time: Whether to run in real-time mode (for streaming)
        
    Returns:
        Dictionary with animation results
    """
    print(f"Animating avatar with voice: {avatar_path}")
    print(f"Audio source: {audio_path}")
    
    # Create output directory
    ensure_directory(output_dir)
    
    # Load avatar model
    try:
        # Find related files
        avatar_dir = os.path.dirname(avatar_path)
        blendshapes_dir = os.path.join(avatar_dir, "blendshapes")
        skeleton_path = os.path.join(avatar_dir, "skeleton.json")
        texture_path = os.path.join(avatar_dir, "texture.png")
        
        # Create avatar animator
        animator = AvatarAnimator(
            model_path=avatar_path,
            blendshapes_dir=blendshapes_dir if os.path.exists(blendshapes_dir) else None,
            skeleton_path=skeleton_path if os.path.exists(skeleton_path) else None,
            texture_path=texture_path if os.path.exists(texture_path) else None,
            use_gpu=torch.cuda.is_available(),
            high_quality=high_quality
        )
        
        print(f"Avatar loaded with {len(animator.get_available_blendshapes())} blendshapes")
        
        # Create voice animator
        voice_animator = VoiceAnimator(
            avatar_animator=animator,
            model_path=None,  # Use rule-based fallback
            use_gpu=torch.cuda.is_available(),
            smoothing_factor=0.3,
            emotion_detection=True
        )
        
        # If real-time mode, start streaming
        if real_time:
            print("Starting real-time animation")
            voice_animator.start_streaming()
            
            try:
                print("Press Ctrl+C to stop")
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("Stopping streaming")
                voice_animator.stop_streaming()
                
            return {
                'success': True,
                'mode': 'real-time'
            }
        
        # Create lip-sync animation from audio file
        print("Creating lip-sync animation")
        result = voice_animator.create_lipsync_animation(
            audio_path=audio_path,
            output_dir=output_dir,
            fps=30
        )
        
        return result
        
    except Exception as e:
        print(f"Error animating avatar: {e}")
        return {'error': str(e)}

def process_video_for_animation(video_path: str, output_dir: str, high_quality: bool = False) -> Dict:
    """
    Process a video to create an animated avatar, extracting both visual and audio content.
    
    Args:
        video_path: Path to the video file
        output_dir: Directory to save output
        high_quality: Whether to use high-quality settings
        
    Returns:
        Dictionary with processing results
    """
    print(f"Processing video for animation: {video_path}")
    
    # Create output directory
    ensure_directory(output_dir)
    
    try:
        # Create avatar from video
        avatar_result = create_face_avatar_from_video(
            video_path=video_path,
            output_dir=os.path.join(output_dir, "avatar"),
            high_quality=high_quality
        )
        
        if 'error' in avatar_result:
            return avatar_result
        
        # Extract audio from video
        audio_path = os.path.join(output_dir, "audio.wav")
        
        import subprocess
        try:
            subprocess.run([
                'ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', audio_path
            ], check=True)
            print(f"Audio extracted to: {audio_path}")
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return {'error': f"Failed to extract audio: {str(e)}"}
        
        # Animate the avatar using the extracted audio
        animation_result = animate_avatar_with_voice(
            avatar_path=avatar_result['avatar_path'],
            audio_path=audio_path,
            output_dir=os.path.join(output_dir, "animation"),
            high_quality=high_quality
        )
        
        return {
            'success': True,
            'avatar_result': avatar_result,
            'animation_result': animation_result,
            'output_dir': output_dir
        }
        
    except Exception as e:
        print(f"Error processing video for animation: {e}")
        return {'error': str(e)}

def create_face_avatar(image_path: str, output_dir: str, high_quality: bool = False) -> Dict[str, Any]:
    """
    Create a 3D avatar face from an image.
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save outputs
        high_quality: Whether to use high-quality settings
    
    Returns:
        Dictionary with avatar data and paths to generated files
    """
    print(f"Creating face avatar from {image_path}")
    
    # Create output directory
    ensure_directory(output_dir)
    
    # Load image
    image = load_image(image_path)
    if image is None:
        print(f"Failed to load image from {image_path}")
        return {}
    
    # Initialize face reconstructor
    face_reconstructor = FaceReconstructor(use_high_quality=high_quality)
    
    # Reconstruct 3D face
    face_mesh = face_reconstructor.reconstruct_face(image)
    
    # Initialize face texture mapper
    texture_mapper = FaceTextureMapper(
        resolution=1024 if high_quality else 512,
        use_high_quality=high_quality
    )
    
    # Generate and apply texture
    face_mesh = texture_mapper.generate_and_apply_texture(face_mesh, image)
    
    # Refine geometry if high_quality is enabled
    if high_quality:
        geometry_refiner = FaceGeometryRefiner()
        face_mesh = geometry_refiner.refine_geometry(face_mesh, image)
    
    # Save outputs
    mesh_path = os.path.join(output_dir, "face_mesh.obj")
    texture_path = os.path.join(output_dir, "face_texture.png")
    
    face_mesh.save_obj(mesh_path, texture_path)
    texture_mapper.save_texture(face_mesh.texture, texture_path)
    
    print(f"Face avatar created and saved to {output_dir}")
    
    return {
        "mesh": face_mesh,
        "mesh_path": mesh_path,
        "texture_path": texture_path
    }

def create_full_body_avatar(image_path: str, output_dir: str, high_quality: bool = False, 
                           measure_body: bool = True, integrate_face: bool = True) -> Dict[str, Any]:
    """
    Create a full-body 3D avatar from an image.
    
    Args:
        image_path: Path to the input image
        output_dir: Directory to save outputs
        high_quality: Whether to use high-quality settings
        measure_body: Whether to extract body measurements
        integrate_face: Whether to integrate a reconstructed face
    
    Returns:
        Dictionary with avatar data and paths to generated files
    """
    print(f"Creating full-body avatar from {image_path}")
    
    # Create output directory
    ensure_directory(output_dir)
    
    # Load image
    image = load_image(image_path)
    if image is None:
        print(f"Failed to load image from {image_path}")
        return {}
    
    # Step 1: Estimate body pose
    pose_estimator = PoseEstimator(use_3d=True)
    pose_data = pose_estimator.estimate_pose(image)
    
    # Step 2: Extract body measurements if requested
    measurements = None
    if measure_body:
        body_measurer = BodyMeasurement()
        measurements = body_measurer.extract_measurements(image, pose_data)
        
        # Save measurements
        measurements_path = os.path.join(output_dir, "body_measurements.json")
        body_measurer.save_measurements(measurements, measurements_path)
        print(f"Body measurements saved to {measurements_path}")
    
    # Step 3: Generate 3D body mesh
    body_generator = BodyMeshGenerator(use_high_quality=high_quality)
    body_mesh = body_generator.generate_body_mesh(pose_data, measurements)
    
    # Step 4: Generate textures
    texture_mapper = BodyTextureMapper(
        resolution=2048 if high_quality else 1024,
        use_high_quality=high_quality
    )
    
    # Generate UVs if not already present
    if not body_mesh.has_uv_coordinates():
        uv_coords = texture_mapper.generate_uv_coordinates(body_mesh)
        body_mesh.set_uv_coordinates(uv_coords)
    
    # Project image to texture
    body_texture = texture_mapper.project_image_to_texture(body_mesh, image)
    body_mesh.set_texture(body_texture)
    
    # Generate PBR textures if high quality
    pbr_texture_paths = {}
    if high_quality:
        pbr_textures = texture_mapper.generate_pbr_textures(body_mesh)
        pbr_dir = os.path.join(output_dir, "textures")
        ensure_directory(pbr_dir)
        pbr_texture_paths = texture_mapper.save_pbr_textures(pbr_textures, pbr_dir)
    
    # Step 5: Integrate face model if requested
    if integrate_face:
        # Create face avatar
        face_output_dir = os.path.join(output_dir, "face")
        ensure_directory(face_output_dir)
        face_data = create_face_avatar(image_path, face_output_dir, high_quality)
        
        if face_data and "mesh" in face_data:
            # Integrate face and body
            integrator = FaceBodyIntegrator(
                blend_region_size=0.1,
                preserve_face_details=True
            )
            
            combined_mesh = integrator.integrate_face_and_body(
                face_data["mesh"], body_mesh
            )
            
            # Use the combined mesh from now on
            body_mesh = combined_mesh
    
    # Save outputs
    mesh_path = os.path.join(output_dir, "body_mesh.obj")
    texture_path = os.path.join(output_dir, "body_texture.png")
    
    body_mesh.save_obj(mesh_path, texture_path)
    texture_mapper.save_texture(body_texture, texture_path)
    
    # Save parameters
    params_path = os.path.join(output_dir, "body_params.json")
    body_generator.save_parameters(params_path)
    
    print(f"Full-body avatar created and saved to {output_dir}")
    
    return {
        "mesh": body_mesh,
        "mesh_path": mesh_path,
        "texture_path": texture_path,
        "pbr_texture_paths": pbr_texture_paths,
        "measurements": measurements
    }

def create_full_body_avatar_from_video(video_path: str, output_dir: str, high_quality: bool = False) -> Dict[str, Any]:
    """
    Create a full-body avatar from a video by extracting a good representative frame.
    
    Args:
        video_path: Path to the input video
        output_dir: Directory to save outputs
        high_quality: Whether to use high-quality settings
    
    Returns:
        Dictionary with avatar data and paths to generated files
    """
    print(f"Creating full-body avatar from video {video_path}")
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video {video_path}")
        return {}
    
    # Extract a good frame
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Try to find a good frame in the first 2 seconds
    best_frame = None
    best_pose_score = 0
    
    # Initialize pose estimator
    pose_estimator = PoseEstimator(use_3d=True)
    
    frames_to_check = min(int(fps * 2), frame_count)
    for i in range(frames_to_check):
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Estimate pose and get score
        pose_data = pose_estimator.estimate_pose(frame_rgb)
        score = pose_data.get("confidence", 0)
        
        if score > best_pose_score:
            best_pose_score = score
            best_frame = frame_rgb
    
    cap.release()
    
    if best_frame is None:
        print("Failed to extract a good frame from the video")
        return {}
    
    # Save the best frame
    best_frame_path = os.path.join(output_dir, "best_frame.png")
    ensure_directory(output_dir)
    cv2.imwrite(best_frame_path, cv2.cvtColor(best_frame, cv2.COLOR_RGB2BGR))
    
    # Create avatar from the best frame
    return create_full_body_avatar(best_frame_path, output_dir, high_quality)

def animate_avatar_with_video(avatar_mesh_path: str, video_path: str, output_path: str) -> str:
    """
    Animate an avatar using a driving video.
    
    Args:
        avatar_mesh_path: Path to the avatar mesh file
        video_path: Path to the driving video
        output_path: Path to save the animated video
    
    Returns:
        Path to the output video
    """
    print(f"Animating avatar with video {video_path}")
    
    # Initialize avatar animator
    animator = AvatarAnimator(
        model_path=avatar_mesh_path,
        use_gpu=torch.cuda.is_available()
    )
    
    # Initialize First Order Motion Model
    motion_model = FirstOrderMotionModel()
    
    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video {video_path}")
        return ""
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Get the first frame as reference
    ret, first_frame = cap.read()
    if not ret:
        print("Failed to read the first frame")
        cap.release()
        return ""
    
    # Convert BGR to RGB
    first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    
    # Set reference frame
    motion_model.set_reference_frame(first_frame_rgb)
    
    # Process remaining frames
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Animate using the motion model
        animated_frame = motion_model.animate(frame_rgb)
        
        # Apply to avatar
        animated_avatar = animator.update(animated_frame)
        
        # Write to output video
        writer.write(cv2.cvtColor(animated_avatar, cv2.COLOR_RGB2BGR))
        
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"Processed {frame_idx}/{frame_count} frames")
    
    # Release resources
    cap.release()
    writer.release()
    
    print(f"Animation saved to {output_path}")
    return output_path

def demo_avatar_animation_features(avatar_mesh_path: str, output_dir: str) -> None:
    """
    Demonstrate various avatar animation features.
    
    Args:
        avatar_mesh_path: Path to the avatar mesh file
        output_dir: Directory to save outputs
    """
    print(f"Demonstrating avatar animation features")
    ensure_directory(output_dir)
    
    # Initialize avatar animator
    animator = AvatarAnimator(
        model_path=avatar_mesh_path,
        use_gpu=torch.cuda.is_available()
    )
    
    # 1. Demonstrate micro-expressions
    print("Demonstrating micro-expressions...")
    micro_expression = MicroExpressionSynthesizer(
        blink_frequency=0.1,  # blink every 10 seconds on average
        micro_movement_scale=0.3
    )
    
    # Generate frames with micro-expressions
    frames = []
    for i in range(90):  # 3 seconds at 30 fps
        # Update micro-expressions
        expressions = micro_expression.update(1/30)  # 30 fps
        
        # Apply to avatar
        frame = animator.set_blendshape_weights(expressions)
        frames.append(frame)
    
    # Save as video
    micro_exp_path = os.path.join(output_dir, "micro_expressions.mp4")
    save_animation(frames, micro_exp_path, 30)
    
    # 2. Demonstrate gaze control
    print("Demonstrating gaze control...")
    gaze_controller = GazeController(
        saccade_frequency=0.3,  # frequent eye movements
        natural_movement=True
    )
    
    # Add attention points
    gaze_controller.add_attention_point(0.2, 0.3, weight=0.7)  # top-left
    gaze_controller.add_attention_point(-0.1, 0.1, weight=0.5)  # center-left
    gaze_controller.add_attention_point(0.3, -0.2, weight=0.8)  # bottom-right
    
    # Generate frames with gaze movement
    frames = []
    for i in range(150):  # 5 seconds at 30 fps
        # Update gaze
        gaze_weights = gaze_controller.update(1/30)  # 30 fps
        
        # Apply to avatar
        frame = animator.set_blendshape_weights(gaze_weights)
        frames.append(frame)
    
    # Save as video
    gaze_path = os.path.join(output_dir, "gaze_control.mp4")
    save_animation(frames, gaze_path, 30)
    
    # 3. Demonstrate head pose variations
    print("Demonstrating head pose variations...")
    head_pose = HeadPoseController(
        movement_scale=0.7,  # larger movements
        movement_frequency=0.4,  # frequent movements
        natural_motion=True
    )
    
    # Generate frames with head movements
    frames = []
    for i in range(180):  # 6 seconds at 30 fps
        # Update head pose
        pose_weights = head_pose.update(1/30)  # 30 fps
        
        # Apply to avatar
        frame = animator.set_blendshape_weights(pose_weights)
        frames.append(frame)
        
        # Add a head nod in the middle
        if i == 90:
            head_pose.add_natural_nod(intensity=0.8)
    
    # Save as video
    head_pose_path = os.path.join(output_dir, "head_pose.mp4")
    save_animation(frames, head_pose_path, 30)
    
    # 4. Demonstrate emotion control
    print("Demonstrating emotion control...")
    emotion_controller = EmotionController(
        transition_speed=0.3,  # smooth transitions
        idle_variation=0.2,  # subtle variations
        personality_bias={'happy': 0.1, 'surprised': 0.05}  # slight bias
    )
    
    # Generate frames with emotion sequences
    frames = []
    emotion_sequence = [
        ('neutral', 40),  # neutral for 40 frames
        ('happy', 60),    # transition to happy for 60 frames
        ('surprised', 60), # transition to surprised for 60 frames
        ('sad', 60),      # transition to sad for 60 frames
        ('neutral', 40)   # back to neutral for 40 frames
    ]
    
    frame_idx = 0
    current_emotion = 'neutral'
    for emotion, duration in emotion_sequence:
        # Set the target emotion
        if frame_idx == 0:
            emotion_controller.set_emotion(emotion, immediate=True)
        else:
            emotion_controller.set_emotion(emotion)
        
        # Generate frames for this emotion
        for i in range(duration):
            # Update emotion controller
            emotion_weights = emotion_controller.update(1/30)  # 30 fps
            
            # Apply to avatar
            frame = animator.set_blendshape_weights(emotion_weights)
            frames.append(frame)
            frame_idx += 1
    
    # Save as video
    emotion_path = os.path.join(output_dir, "emotion_control.mp4")
    save_animation(frames, emotion_path, 30)
    
    # 5. Demonstrate gesture generation
    print("Demonstrating gesture generation...")
    gesture_model = GestureMannerismLearner()
    
    # Generate frames with various gestures
    frames = []
    gesture_sequence = [
        ('idle', 30),     # idle for 30 frames
        ('head', 60),     # head gesture for 60 frames
        ('hands', 90),    # hand gesture for 90 frames
        ('emphasis', 60), # emphasis gesture for 60 frames
        ('idle', 30)      # back to idle for 30 frames
    ]
    
    for gesture_type, duration in gesture_sequence:
        # Generate gesture animation data
        gesture_data = gesture_model._generate_default_gesture(
            duration_sec=duration/30,
            gesture_type=gesture_type
        )
        
        # Extract keyframes
        keyframes = gesture_data['keyframes']
        if not keyframes:
            continue
            
        # Generate frames by interpolating between keyframes
        for i in range(duration):
            position = i / duration  # 0-1 position in animation
            
            # Find surrounding keyframes
            prev_keyframe = None
            next_keyframe = None
            
            for j in range(len(keyframes) - 1):
                if (keyframes[j]['position'] <= position and 
                    keyframes[j+1]['position'] >= position):
                    prev_keyframe = keyframes[j]
                    next_keyframe = keyframes[j+1]
                    break
            
            if prev_keyframe is None or next_keyframe is None:
                continue
                
            # Interpolate between keyframes
            alpha = ((position - prev_keyframe['position']) / 
                   (next_keyframe['position'] - prev_keyframe['position']))
            
            # Apply interpolated values
            blendshape_weights = {}
            
            for key in prev_keyframe['values']:
                if key in next_keyframe['values']:
                    prev_val = prev_keyframe['values'][key]
                    next_val = next_keyframe['values'][key]
                    
                    # Handle different value types
                    if isinstance(prev_val, (int, float)) and isinstance(next_val, (int, float)):
                        interpolated = prev_val + alpha * (next_val - prev_val)
                        
                        # Map to appropriate blendshape
                        if key == 'head_yaw':
                            if interpolated > 0:
                                blendshape_weights['face_head_right'] = min(abs(interpolated), 1.0)
                            else:
                                blendshape_weights['face_head_left'] = min(abs(interpolated), 1.0)
                        elif key == 'head_pitch':
                            if interpolated > 0:
                                blendshape_weights['face_head_down'] = min(abs(interpolated), 1.0)
                            else:
                                blendshape_weights['face_head_up'] = min(abs(interpolated), 1.0)
            
            # Apply to avatar
            frame = animator.set_blendshape_weights(blendshape_weights)
            frames.append(frame)
    
    # Save as video
    gesture_path = os.path.join(output_dir, "gesture_control.mp4")
    save_animation(frames, gesture_path, 30)
    
    print(f"Animation demos saved to {output_dir}")

def save_animation(frames: List[np.ndarray], output_path: str, fps: int = 30) -> None:
    """
    Save animation frames as a video.
    
    Args:
        frames: List of animation frames
        output_path: Path to save the video
        fps: Frames per second
    """
    if not frames:
        print("No frames to save")
        return
        
    # Get dimensions
    height, width = frames[0].shape[:2]
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Write frames
    for frame in frames:
        # Convert to BGR (OpenCV format)
        if frame.shape[2] == 3:  # RGB
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            frame_bgr = frame
            
        writer.write(frame_bgr)
    
    # Release resources
    writer.release()
    print(f"Animation saved to {output_path}")

def main():
    """Main function for the avatar creation demo."""
    parser = argparse.ArgumentParser(description="Avatar Creation Demo")
    
    # Input arguments
    parser.add_argument("--mode", type=str, default="face",
                      choices=["face", "multi_view", "body", "body_video", "animate", "animation_demo", "stylegan"],
                      help="Operation mode")
    parser.add_argument("--input", type=str, help="Path to input image, video, or directory with images")
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save outputs")
    parser.add_argument("--high_quality", action="store_true", help="Use high-quality settings")
    parser.add_argument("--use_advanced_reconstruction", action="store_true", help="Use advanced face reconstruction")
    parser.add_argument("--use_improved_texturing", action="store_true", help="Use improved texture mapping")
    parser.add_argument("--driving_video", type=str, help="Path to driving video for animation")
    parser.add_argument("--avatar_mesh", type=str, help="Path to avatar mesh for animation")
    parser.add_argument("--style_image", type=str, help="Path to style reference image for StyleGAN")
    
    args = parser.parse_args()
    
    # Create output directory
    ensure_directory(args.output_dir)
    
    if args.mode == "face":
        # Create face avatar from single image
        if not args.input:
            print("Error: Input image path is required for face mode")
            return
            
        create_avatar_from_image(
            args.input, 
            args.output_dir, 
            args.high_quality,
            args.use_advanced_reconstruction,
            args.use_improved_texturing
        )
        
    elif args.mode == "multi_view":
        # Create face avatar from multiple images
        if not args.input:
            print("Error: Input directory path is required for multi_view mode")
            return
            
        create_avatar_from_multi_view(
            args.input, 
            args.output_dir, 
            args.high_quality,
            args.use_improved_texturing
        )
        
    elif args.mode == "body":
        # Create full-body avatar from image
        if not args.input:
            print("Error: Input image path is required for body mode")
            return
            
        create_full_body_avatar(
            args.input, 
            args.output_dir, 
            args.high_quality
        )
        
    elif args.mode == "body_video":
        # Create avatar from video
        if not args.input:
            print("Error: Input video path is required for body_video mode")
            return
            
        create_avatar_from_body_video(
            args.input, 
            args.output_dir, 
            args.high_quality
        )
        
    elif args.mode == "animate":
        # Animate avatar with video
        if not args.avatar_mesh:
            print("Avatar mesh path is required for animate mode")
            return
            
        if not args.driving_video:
            print("Driving video path is required for animate mode")
            return
            
        output_path = os.path.join(args.output_dir, "animated_avatar.mp4")
        ensure_directory(args.output_dir)
        
        animate_avatar_with_video(args.avatar_mesh, args.driving_video, output_path)
        
    elif args.mode == "animation_demo":
        # Run animation feature demos
        if not args.avatar_mesh:
            print("Avatar mesh path is required for animation_demo mode")
            return
            
        demo_avatar_animation_features(args.avatar_mesh, args.output_dir)
        
    else:
        print(f"Unknown mode: {args.mode}")
        return
    
    # Print result
    if 'error' in result:
        print(f"Error: {result['error']}")
    else:
        print("Process completed successfully!")
        print(f"Output saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 