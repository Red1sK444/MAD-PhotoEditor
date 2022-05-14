package com.example.android.wonderfulapp

import android.Manifest
import android.app.AlertDialog
import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.view.MenuItem
import android.view.View
import android.widget.HorizontalScrollView
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.result.ActivityResultLauncher
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.FileProvider
import androidx.fragment.app.Fragment
import by.kirich1409.viewbindingdelegate.viewBinding
import com.example.android.wonderfulapp.contracts.OpenGalleryContract
import com.example.android.wonderfulapp.contracts.SavePictureContract
import com.example.android.wonderfulapp.databinding.ActivityMainBinding
import com.example.android.wonderfulapp.figure.FigureFragment
import com.example.android.wonderfulapp.image.ImageMainFragment
import com.example.android.wonderfulapp.image.ImageManager
import com.example.android.wonderfulapp.spline.SplineFragment
import java.io.File
import java.io.IOException
import java.text.SimpleDateFormat
import java.util.*

class MainActivity : AppCompatActivity(R.layout.activity_main) {

    private val viewBinding by viewBinding(ActivityMainBinding::bind, R.id.rootLayout)
    private val fragmentManagerUtil =
        FragmentManagerUtil(supportFragmentManager, R.id.fragmentContainer)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        viewBinding.toolbar.inflateMenu(R.menu.top_action_menu)
        viewBinding.toolbar.setOnMenuItemClickListener { item ->
            performMenuAction(item)
        }
        viewBinding.bottomNavigationView.setOnNavigationItemSelectedListener { item ->
            performBottomNavigation(item)
        }

        fragmentManagerUtil.navigateToFragment(
            ImageMainFragment.TAG,
            ImageMainFragment.newInstance()
        )
    }

    private fun performMenuAction(item: MenuItem): Boolean {
        return when (item.itemId) {
            R.id.open_photo -> {
                selectImage(this)
                true
            }
            R.id.share_photo -> {
                processPictureAction(Manifest.permission.WRITE_EXTERNAL_STORAGE, savePictureHandler)
                true
            }
            else -> {
                super.onOptionsItemSelected(item)
            }
        }
    }

    private fun performBottomNavigation(item: MenuItem): Boolean {
        return when (item.itemId) {
            R.id.photo_fragment -> {
                selectScreen(ImageMainFragment.TAG, ImageMainFragment.newInstance())
                true
            }
            R.id.spline_fragment -> {
                selectScreen(SplineFragment.TAG, SplineFragment.newInstance())
                true
            }
            R.id.figure_fragment -> {
                selectScreen(FigureFragment.TAG, FigureFragment.newInstance())
                true
            }
            else -> false
        }
    }

    private fun selectScreen(tag: String, fragment: Fragment) {
        fragmentManagerUtil.navigateToFragment(tag, fragment)
        updateActionBar(tag)
    }

    private fun updateActionBar(tag: String) {
        when (tag) {
            ImageMainFragment.TAG -> showPhotoActions()
            SplineFragment.TAG -> showSplineActions()
            FigureFragment.TAG -> showFigureActions()
        }
    }

    private fun showPhotoActions() {
        showMenuItem(viewBinding.toolbar.menu.findItem(R.id.open_photo))
        showMenuItem(viewBinding.toolbar.menu.findItem(R.id.share_photo))
    }

    private fun showSplineActions() {
        hideMenuItem(viewBinding.toolbar.menu.findItem(R.id.open_photo))
        hideMenuItem(viewBinding.toolbar.menu.findItem(R.id.share_photo))
    }

    private fun showFigureActions() {
        hideMenuItem(viewBinding.toolbar.menu.findItem(R.id.open_photo))
        hideMenuItem(viewBinding.toolbar.menu.findItem(R.id.share_photo))
    }

    private fun hideMenuItem(item: MenuItem) {
        item.isEnabled = false
        item.isVisible = false
    }

    private fun showMenuItem(item: MenuItem) {
        item.isEnabled = true
        item.isVisible = true
    }

//--------------------------------------------------------------------------------------------------

    companion object {
        private const val MIMETYPE_IMAGES = "image/*"
    }

    private lateinit var currentPhotoPath: Uri

    private val openCameraHandler =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            when {
                granted -> {
                    dispatchTakePicture()
                }
                !shouldShowRequestPermissionRationale(Manifest.permission.CAMERA) -> {
                    Toast.makeText(
                        this,
                        getString(R.string.do_not_ask_again_option_on),
                        Toast.LENGTH_LONG
                    ).show()
                }
                else -> {
                    Toast.makeText(
                        this,
                        getString(R.string.camera_permission_denied),
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
        }

    private val openGalleryHandler =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            when {
                granted -> {
                    dispatchOpenGallery()
                }
                !shouldShowRequestPermissionRationale(Manifest.permission.READ_EXTERNAL_STORAGE) -> {
                    Toast.makeText(
                        this,
                        getString(R.string.do_not_ask_again_option_on),
                        Toast.LENGTH_LONG
                    ).show()
                }
                else -> {
                    Toast.makeText(
                        this,
                        getString(R.string.storage_permission_denied),
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
        }

    private val savePictureHandler =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            when {
                granted -> {
                    if (ImageManager.image != null) {
                        dispatchSavePicture()
                    } else {
                        Toast.makeText(
                            this,
                            getString(R.string.picture_not_found),
                            Toast.LENGTH_SHORT
                        )
                            .show()
                    }
                }
                !shouldShowRequestPermissionRationale(Manifest.permission.WRITE_EXTERNAL_STORAGE) -> {
                    Toast.makeText(
                        this,
                        getString(R.string.do_not_ask_again_option_on),
                        Toast.LENGTH_LONG
                    ).show()
                }
                else -> {
                    Toast.makeText(
                        this,
                        getString(R.string.storage_permission_denied),
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
        }

    private val takeCameraPictureLauncher =
        registerForActivityResult(ActivityResultContracts.TakePicture()) { result ->
            if (result) {
                setPicFromCurPath()
            }
        }

    private val takeGalleryPictureLauncher =
        registerForActivityResult(OpenGalleryContract()) { imageUri ->
            showSelectedPhoto(imageUri)
        }

    private val savePictureLauncher =
        registerForActivityResult(SavePictureContract()) { saveUri ->
            saveUri?.let { savePhoto(it) }
        }

    private fun selectImage(context: Context) {
        val options = arrayOf(
            getString(R.string.dialog_take_photo),
            getString(R.string.dialog_open_photo),
            getString(R.string.dialog_dismiss)
        )

        val builder: AlertDialog.Builder = AlertDialog.Builder(context)

        builder.setItems(options) { dialog, item ->
            when (options[item]) {
                getString(R.string.dialog_take_photo) -> {
                    processPictureAction(Manifest.permission.CAMERA, openCameraHandler)
                }
                getString(R.string.dialog_open_photo) -> {
                    processPictureAction(
                        Manifest.permission.READ_EXTERNAL_STORAGE,
                        openGalleryHandler
                    )
                }
                getString(R.string.dialog_dismiss) -> {
                    dialog.dismiss()
                }
            }
        }
        builder.show()
    }

    private fun processPictureAction(permission: String, handler: ActivityResultLauncher<String>) {
        handler.launch(permission)
    }

    private fun dispatchTakePicture() {
        val photoFile: File? = try {
            createImageFile()
        } catch (ex: IOException) {
            Toast.makeText(this, getString(R.string.taking_photo_error), Toast.LENGTH_LONG)
                .show()
            null
        }

        photoFile?.let {
            val photoURI: Uri = FileProvider.getUriForFile(
                this,
                "com.example.android.wonderfulapp",
                it
            )
            takeCameraPictureLauncher.launch(photoURI)
        }
    }

    @Throws(IOException::class)
    private fun createImageFile(): File {
        val timeStamp: String = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val storageDir: File? = this.getExternalFilesDir(Environment.DIRECTORY_PICTURES)
        return File.createTempFile(
            "JPEG_${timeStamp}_",
            ".jpg",
            storageDir
        ).apply {
            currentPhotoPath = FileProvider.getUriForFile(
                this@MainActivity,
                "com.example.android.wonderfulapp",
                this
            )
        }
    }

    private fun dispatchOpenGallery() {
        takeGalleryPictureLauncher.launch(MIMETYPE_IMAGES)
    }

    private fun dispatchSavePicture() {
        val timeStamp: String = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val imageName = "JPEG_${timeStamp}_"

        savePictureLauncher.launch(imageName)
    }

    private fun savePhoto(saveUri: Uri) {
        val outputStream = contentResolver.openOutputStream(saveUri)
        try {
            ImageManager.image?.compress(Bitmap.CompressFormat.JPEG, 100, outputStream)
            outputStream?.let {
                it.flush()
                it.close()
            }
            Toast.makeText(this, getString(R.string.picture_saved), Toast.LENGTH_SHORT).show()
        } catch (e: Exception) {
            Toast.makeText(this, getString(R.string.save_error), Toast.LENGTH_LONG)
                .show()
        }
    }

    private fun showSelectedPhoto(selectedImage: Uri?) {

        selectedImage?.let {
            currentPhotoPath = it
            setPicFromCurPath()
        }
    }

    private fun setPicFromCurPath() {
        fragmentManagerUtil.navigateToFragment(
            ImageMainFragment.TAG,
            ImageMainFragment.newInstance()
        )

        val fragmentImageView =
            supportFragmentManager.findFragmentByTag(ImageMainFragment.TAG)?.view?.findViewById<ImageView>(
                R.id.imageView
            )

        fragmentImageView?.let {
            ImageManager.updateImage(currentPhotoPath, this.contentResolver)
            it.setImageBitmap(ImageManager.thumbnail)

            showAlgorithmList()
        }
    }

    private fun showAlgorithmList() {
        val fragmentAlgorithmScrollView =
            supportFragmentManager.findFragmentByTag(ImageMainFragment.TAG)?.view?.findViewById<HorizontalScrollView>(
                R.id.algorithmScrollView
            )

        fragmentAlgorithmScrollView?.let { view ->
            view.visibility = View.VISIBLE
        }
    }
}