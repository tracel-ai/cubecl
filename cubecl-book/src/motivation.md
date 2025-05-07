# Why CubeCL is Important

In the current landscape of high-performance computing, developers face several significant
challenges that CubeCL aims to address:

### Complexity in Performance Optimization

Optimizing compute kernels for different hardware is inherently complex. Developers must understand
the intricacies of each platform’s architecture and API, leading to a steep learning curve and the
risk of suboptimal performance. The need for manual tuning and platform-specific adjustments often
results in code that is difficult to maintain and extend.

The simplest way to solve this problem is to provide high-level abstractions that can be composed in
a variety of ways. All of those variations can be autotuned to select the best settings for the
current hardware and problem at hand.

### Lack of Portability

Portability remains a significant issue. Code written for one API or even for a single GPU
architecture often cannot be easily transferred to another, hindering the ability to develop
software that can run across diverse hardware environments.

The GPU computing ecosystem is fragmented, with different hardware platforms like CUDA, ROCm, Metal,
and Vulkan requiring their own specialized codebases. This fragmentation forces developers to write
and maintain separate implementations for each platform, increasing both development time and
complexity.

## Conclusion

In essence, these challenges underscore the need for a more unified and developer-friendly approach
to high-performance computing. CubeCL seeks to bridge these gaps by addressing the core issues
within the current ecosystem, offering a new direction for high-performance and portable computing
solutions.
