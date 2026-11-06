"""Public empirical SQR step policy; does not inherit Huber convergence theory."""
import numpy as np


def lambda1_step(c, second_moment_upper):
    """EXACT eta=c/(2 lambda1): no h, kernel, sample or error-density input."""
    if not all(np.isfinite(v) and v>0 for v in (c,second_moment_upper)):
        raise ValueError('step_c and public_second_moment_upper must be positive and finite')
    return float(c/(2*second_moment_upper))


def resolve_step(options):
    """Consume explicit public policy fields, leaving method-compatible kwargs."""
    options=dict(options)
    if options.get('eta') != 'public_lambda1':
        if 'step_c' in options or 'public_second_moment_upper' in options:
            raise ValueError('step_c/public_second_moment_upper require eta=public_lambda1')
        return options,{}
    c=options.pop('step_c')
    upper=options.pop('public_second_moment_upper')
    eta=lambda1_step(c,upper)
    options['eta']=eta
    return options,dict(step_policy='eta=c/(2*lambda1)',step_size=eta,
                        step_c=c,public_second_moment_upper=upper,
                        step_uses_bandwidth=False,step_uses_kernel_factor=False)
