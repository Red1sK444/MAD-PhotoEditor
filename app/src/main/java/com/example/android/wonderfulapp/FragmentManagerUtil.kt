package com.example.android.wonderfulapp

import androidx.fragment.app.Fragment
import androidx.fragment.app.FragmentManager
import androidx.fragment.app.commit

class FragmentManagerUtil(
    private val fragmentManager: FragmentManager,
    private val containerId: Int,
) {
    fun navigateToFragment(tag: String, fragment: Fragment) {
        fragmentManager.commit {
            val active = findActiveFragment()
            val target = fragmentManager.findFragmentByTag(tag)

            if (active != null && target != null && active == target) return@commit

            if (active != null) {
                hide(active)
            }

            if (target == null) {
                add(containerId, fragment, tag)
            } else {
                show(target)
            }
        }
    }

    private fun findActiveFragment() = fragmentManager.fragments.find { it.isVisible }
}