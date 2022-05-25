# See the License for the specific language governing permissions and
# limitations under the License.
 
set -e 
current_folder="$( cd "$(dirname "$0")" ;pwd -P )"


SAMPLE_FOLDER=(
/plugin/RoadSegPostProcess/
)


err_flag=0
for sample in "${SAMPLE_FOLDER[@]}";do
    cd "${current_folder}/${sample}"
    bash build.sh || {
        echo -e "Failed to build ${sample}"
		err_flag=1
    }
done

if [ ${err_flag} -eq 1 ]; then
	exit 1
fi
exit 0
