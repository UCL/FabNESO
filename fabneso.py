try:
    from fabsim.base.fab import add_local_paths
except ImportError:
    from base.fab import add_local_paths

# Add local script, blackbox and template path.
add_local_paths("fabneso")
