import pysam

unmapped_reads = 0
total_reads = 0
num_zero_quality_reads = 0
map_quality_sum = 0
avg_map_quality = 0
avg_map_quality_non_zero = 0

bam_file = pysam.AlignmentFile(r"../gi-2021-etf/exercise/dz2/merged-tumor.bam")

for read in bam_file:
    total_reads += 1

    if read.is_unmapped:
        unmapped_reads += 1

    if read.mapping_quality == 0:
        num_zero_quality_reads += 1

    map_quality_sum += read.mapping_quality

print(r"Total number of reads: " + str(total_reads))
print(r"Total number of unmapped reads: " + str(unmapped_reads))
print(r"Total number of reads with quality equal to 0: " + str(num_zero_quality_reads))
avg_map_quality_non_zero = map_quality_sum / (total_reads - num_zero_quality_reads)
avg_map_quality = map_quality_sum / total_reads
print(r"Average mapping quality for all the reads: " + str(avg_map_quality))
print(r"Average mapping quality for all reads except zero-quality reads: " + str(avg_map_quality_non_zero))
